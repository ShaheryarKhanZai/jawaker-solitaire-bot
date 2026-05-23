import cv2
import json
import os
import numpy as np
import time
import pyautogui
import keyboard
import mss
import copy
import math
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import threading
from PyQt5 import QtWidgets, QtCore

print("Parent Folder: ",os.getcwd())
crops_dir="symbols"
# ---------------- LOAD SYMBOL TEMPLATES ----------------
templates = {}
for f in os.listdir(crops_dir):
    if f.lower().endswith((".jpg", ".png")):
        tpl_name = os.path.splitext(f)[0].strip()
        base_name = tpl_name.split("_")[0]  # multiple templates per base name
        tpl_img = cv2.imread(os.path.join(crops_dir, f), cv2.IMREAD_COLOR)
        if tpl_img is not None:
            templates.setdefault(base_name, []).append(tpl_img)
template_thresholds = {
    # default threshold for anything else

    'default': 0.85,
    'd': 0.8,
}

def correct_tableau_order(game_state):
    RANKS = {
        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13,
    }

    def rank(card):
        return RANKS[card[:-1]]

    fixed = []
    for pile in game_state["tablaue"]:
        if not pile:
            fixed.append([])
            continue

        # separate fd and face-up
        if pile[0] == "fd":
            fd_cards = []
            idx = 0
            while idx < len(pile) and pile[idx] == "fd":
                fd_cards.append("fd")
                idx += 1
            face_up = pile[idx:]
        else:
            fd_cards = []
            face_up = pile[:]

        # Sort the face-up cards in descending order
        if face_up:
            # sort by rank descending
            face_up_sorted = sorted(face_up, key=rank, reverse=True)
            fixed.append(fd_cards + face_up_sorted)
        else:
            fixed.append(fd_cards)

    game_state["tablaue"] = fixed
    return game_state

def compute_crop_roi(image, regions_json):
    h, w = image.shape[:2]

    with open(regions_json, "r") as f:
        regions = json.load(f)

    # start with extremes
    min_x = 1.0
    min_y = 1.0
    max_x = 0.0
    max_y = 0.0

    for r in regions.values():
        min_x = min(min_x, r["x1"])
        min_y = min(min_y, r["y1"])
        max_x = max(max_x, r["x2"])
        max_y = max(max_y, r["y2"])

    # convert to pixel coordinates
    x1 = int(min_x * w)
    y1 = int(min_y * h)
    x2 = int(max_x * w)
    y2 = int(max_y * h)

    # clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return x1, y1, x2, y2

def detect_solitaire_cards(img, templates, x1, y1, x2, y2, scales=[0.975,1.0,1.025],
                           regions_json="regions.json",
                           template_thresholds=template_thresholds,
                           nms_threshold=0.3, num_workers=8,
                           detection_mask=None, prev_game_state=None, prev_cards_coords=None):
    import cv2, json, numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if img is None:
        raise FileNotFoundError("Input image not found")

    if template_thresholds is None:
        template_thresholds = {"default": 0.85}

    # ---------------- CROP IMAGE TO AREA OF INTEREST ----------------
    full_h, full_w = img.shape[:2]
    
    crop_x1 = max(0, x1) 
    crop_y1 = max(0, y1)
    crop_x2 = min(full_w, x2) 
    crop_y2 = min(full_h, y2)
    img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    img_crop = cv2.resize(img_crop, (461, 580))

    # ---------------- BUILD GREEN MASK ON COLOR IMAGE ----------------
    # Do this before grayscale conversion — HSV needs BGR input
    LOWER_GREEN = np.array([30, 30, 30])
    UPPER_GREEN = np.array([85, 255, 255])

    img_crop_hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(img_crop_hsv, LOWER_GREEN, UPPER_GREEN)
    # green_mask: 255 where green, 0 elsewhere
    # non_green_mask: 255 where NOT green — used to zero out green pixels
    non_green_mask = cv2.bitwise_not(green_mask)

    # ---------------- GRAYSCALE + APPLY GREEN MASK ----------------
    #img_crop_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    # Zero out green pixels in the grayscale image
    # Template matching will now see 0 (black) in green areas — effectively ignored
    img_crop_masked = cv2.bitwise_and(img_crop, img_crop, mask=non_green_mask)

    H, W = img_crop_masked.shape[:2]

    # ---------------- LOAD REGIONS ----------------
    with open(regions_json, "r") as f:
        regions = json.load(f)

    # ---------------- MASK HELPER ----------------
    def should_detect(region_name):
        if detection_mask is None:
            return True

        if region_name.startswith("t"):
            idx = int(region_name[1:])
            mask_list = detection_mask.get("tablaue", [])
            if idx < len(mask_list):
                return bool(mask_list[idx])
            return True
        
        elif region_name.startswith("f"):
            idx = int(region_name[1:])
            mask_list = detection_mask.get("foundation", [])
            if idx < len(mask_list):
                return bool(mask_list[idx])
            return True
        
        elif region_name == "stock":
            mask_list = detection_mask.get("stock", [1])
            return bool(mask_list[0])
        
        elif region_name == "waste":
            mask_list = detection_mask.get("waste", [1])
            return bool(mask_list[0])
        
        return True

    denorm_regions = {}
    for name, r in regions.items():
        dx1 = int(r["x1"] * W)
        dy1 = int(r["y1"] * H)
        dx2 = int(r["x2"] * W)
        dy2 = int(r["y2"] * H)
        denorm_regions[name] = (dx1, dy1, dx2, dy2)

    # ---------------- NMS FUNCTION ----------------
    def nms(boxes, scores, threshold):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes, dtype=float)
        scores = np.array(scores, dtype=float)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        return keep

    # ---------------- FD DETECTION VIA RED STRIP ----------------
    # For t0–t6: check top 20 rows of the region in HSV.
    # If the majority of non-green pixels are red → mark as facedown (fd).
    # Red wraps around in HSV hue: 0–10 and 160–180.
    LOWER_RED_1 = np.array([0,   70, 50])
    UPPER_RED_1 = np.array([10, 255, 255])
    LOWER_RED_2 = np.array([160, 70, 50])
    UPPER_RED_2 = np.array([180, 255, 255])

    fd_tableau = set()
    for region_name, (rx1, ry1, rx2, ry2) in denorm_regions.items():
        if not region_name.startswith("t"):
            continue

        # Clamp the 20px strip to region bounds
        strip_y2 = min(ry1 + 20, ry2)
        hsv_strip = img_crop_hsv[ry1:strip_y2, rx1:rx2]

        if hsv_strip.size == 0:
            continue

        # Exclude green pixels from the strip before counting red
        green_strip = cv2.inRange(hsv_strip, LOWER_GREEN, UPPER_GREEN)
        red_mask_1  = cv2.inRange(hsv_strip, LOWER_RED_1, UPPER_RED_1)
        red_mask_2  = cv2.inRange(hsv_strip, LOWER_RED_2, UPPER_RED_2)
        red_mask    = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Only count red pixels that are not green
        red_not_green   = cv2.bitwise_and(red_mask, cv2.bitwise_not(green_strip))
        total_not_green = int(np.sum(green_strip == 0))

        if total_not_green > 0:
            red_ratio = int(np.sum(red_not_green > 0)) / total_not_green
            if red_ratio > 0.3:   # >30% of visible pixels are red → fd card present
                fd_tableau.add(region_name)

    # ---------------- THREAD FUNCTION PER REGION ----------------
    # Uses img_crop_masked so template matching skips green areas (they are zeroed)
    def process_region(region_name, rx1, ry1, rx2, ry2, scales):
        region_green_mask = green_mask[ry1:ry2, rx1:rx2]
        total_pixels = region_green_mask.size
        if total_pixels > 0:
            green_ratio = np.sum(region_green_mask > 0) / total_pixels
            if green_ratio > 0.85:  # >85% green → nothing useful here, skip
                return region_name, []
        gray_crop = img_crop_masked[ry1:ry2, rx1:rx2]
        region_boxes = []

        for base_name, tpl_list in templates.items():
            # Skip the fd template entirely — fd is now detected via red strip
            if base_name == "fd":
                continue

            t_thresh = template_thresholds.get(base_name, template_thresholds['default'])
            
            for tpl in tpl_list:
                # Convert template to grayscale if needed
                tpl_gray = tpl

                for s in scales:
                    tpl_scaled = cv2.resize(
                        tpl_gray, None, fx=s, fy=s,
                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
                    )
                    th, tw = tpl_scaled.shape[:2]
                    if th > gray_crop.shape[0] or tw > gray_crop.shape[1]:
                        continue

                    res = cv2.matchTemplate(gray_crop, tpl_scaled, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= t_thresh)
                    scores = res[loc].tolist()

                    boxes_coords = [
                        (x + rx1, y + ry1, x + tw + rx1, y + th + ry1, base_name)
                        for x, y in zip(*loc[::-1])
                    ]

                    coords_only = [(bx1, by1, bx2, by2) for bx1, by1, bx2, by2, _ in boxes_coords]
                    keep_indices = nms(coords_only, scores, nms_threshold)

                    for i in keep_indices:
                        x1b, y1b, x2b, y2b, name = boxes_coords[i]
                        region_boxes.append((x1b, y1b, x2b, y2b, name))

        return region_name, region_boxes

    # ---------------- RUN ALL REGIONS IN THREADS ----------------
    results = {}
    regions_to_skip = [rn for rn in denorm_regions if not should_detect(rn)]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_region, rn, *denorm_regions[rn], scales)
            for rn in denorm_regions
            if should_detect(rn)
        ]
        for fut in as_completed(futures):
            region_name, boxes = fut.result()
            results[region_name] = boxes

    # ---------------- RANK/SUIT PAIRING ----------------
    RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    SUITS = ['c','s','h','d']

    cards_coords = {}
    game_state = {"foundation": [], "stock": [], "tablaue": [[] for _ in range(7)], "waste": []}

    # ---------------- COPY PREVIOUS STATE FOR SKIPPED REGIONS ----------------
    if prev_game_state is not None and prev_cards_coords is not None:
        for rn in regions_to_skip:
            results[rn] = []

            if rn.startswith("t"):
                idx = int(rn[1:])
                game_state["tablaue"][idx] = prev_game_state["tablaue"][idx][:]

            elif rn.startswith("f"):
                idx = int(rn[1:])
                if idx < len(prev_game_state["foundation"]):
                    game_state["foundation"].append(prev_game_state["foundation"][idx])

            elif rn == "stock":
                game_state["stock"] = prev_game_state["stock"][:]

            elif rn == "waste":
                game_state["waste"] = prev_game_state["waste"][:]

            for card, coords in prev_cards_coords.items():
                if rn.startswith("t"):
                    idx = int(rn[1:])
                    pile = prev_game_state["tablaue"][idx]
                    if card in pile:
                        cards_coords[card] = coords
                elif rn.startswith("f"):
                    idx = int(rn[1:])
                    if idx < len(prev_game_state["foundation"]):
                        if card == prev_game_state["foundation"][idx]:
                            cards_coords[card] = coords
                elif rn == "stock":
                    if card in prev_game_state["stock"]:
                        cards_coords[card] = coords
                elif rn == "waste":
                    if card in prev_game_state["waste"]:
                        cards_coords[card] = coords

    for region_name, region_boxes in results.items():
        boxes_ranks = [(x1, y1, x2, y2, n) for x1, y1, x2, y2, n in region_boxes if n in RANKS]
        boxes_suits = [(x1, y1, x2, y2, n) for x1, y1, x2, y2, n in region_boxes if n in SUITS]

        added_cards = set()
        for rx1b, ry1b, rx2b, ry2b, rname in boxes_ranks:
            rcx, rcy = (rx1b+rx2b)//2, (ry1b+ry2b)//2
            best_suit = None
            min_dx = float('inf')
            for sx1b, sy1b, sx2b, sy2b, sname in boxes_suits:
                scx, scy = (sx1b+sx2b)//2, (sy1b+sy2b)//2
                dx = scx - rcx
                dy = abs(scy - rcy)
                if dx > 0 and dy <= 15:
                    if dx < min_dx:
                        min_dx = dx
                        best_suit = sname

            if best_suit:
                card_name = rname + best_suit
                if card_name not in added_cards:
                    added_cards.add(card_name)
                    full_cx = int(rcx * (crop_x2-crop_x1)/W + crop_x1)
                    full_cy = int(rcy * (crop_y2-crop_y1)/H + crop_y1)
                    cards_coords[card_name] = {"cx": full_cx, "cy": full_cy}

                    if region_name.startswith("f"):
                        game_state["foundation"].append(card_name)
                    elif region_name.startswith("t"):
                        idx = int(region_name[1:])
                        game_state["tablaue"][idx].append(card_name)
                    elif region_name == "stock":
                        game_state["stock"].append(card_name)
                    elif region_name == "waste":
                        game_state["waste"].append(card_name)

    # ---------------- ADD REGION SLOT CENTERS ----------------
    for i in range(4):
        rx1b, ry1b, rx2b, ry2b = denorm_regions[f"f{i}"]
        full_cx = int((rx1b+rx2b)//2 * (crop_x2-crop_x1)/W + crop_x1)
        full_cy = int((ry1b+ry2b)//2 * (crop_y2-crop_y1)/H + crop_y1)
        cards_coords[f"f_{i}"] = {"cx": full_cx, "cy": full_cy}

    for i in range(7):
        rx1b, ry1b, rx2b, ry2b = denorm_regions[f"t{i}"]
        top_y = int(ry1b + 0.3*(ry2b-ry1b))
        full_cx = int((rx1b+rx2b)//2 * (crop_x2-crop_x1)/W + crop_x1)
        full_cy = int(top_y * (crop_y2-crop_y1)/H + crop_y1)
        cards_coords[f"t_{i}"] = {"cx": full_cx, "cy": full_cy}

    rx1b, ry1b, rx2b, ry2b = denorm_regions["stock"]
    full_cx = int((rx1b+rx2b)//2 * (crop_x2-crop_x1)/W + crop_x1)
    full_cy = int((ry1b+ry2b)//2 * (crop_y2-crop_y1)/H + crop_y1)
    cards_coords[f"stock_0"] = {"cx": full_cx, "cy": full_cy}

    # ---------------- REVERSE TABLEAUX ----------------
    for i in range(7):
        if f"t{i}" not in regions_to_skip:
            game_state["tablaue"][i] = game_state["tablaue"][i][::-1]
            # Use red-strip fd detection result instead of fd template
            if f"t{i}" in fd_tableau:
                game_state["tablaue"][i].insert(0, "fd")

    # ---------------- MERGE PREV COORDS FOR SKIPPED REGIONS ----------------
    if prev_cards_coords is not None and regions_to_skip:
        for card, coords in prev_cards_coords.items():
            if card not in cards_coords:
                cards_coords[card] = coords

    return game_state, cards_coords
def get_best_move(game_state, moves_list, depth, empty_top=None):
    import copy

    gs = copy.deepcopy(game_state)

    if empty_top is None:
        empty_top = {}
    # force to dict if accidentally passed as list
    if not isinstance(empty_top, dict):
        empty_top = {}

    if len(gs['foundation']) < 4:
        gs['foundation'] += [None] * (4 - len(gs['foundation']))

    RANK = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
            '8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
    RED = {'h','d'}
    BLACK = {'s','c'}

    def is_red(s): return s in RED
    def is_black(s): return s in BLACK

    def has_fd(pile):
        return bool(pile) and pile[0] == "fd"

    def first_faceup(pile):
        return 1 if has_fd(pile) else 0

    def can_foundation(card, fcard):
        r, s = card[:-1], card[-1]
        if fcard is None:
            return r == 'A'
        fr, fs = fcard[:-1], fcard[-1]
        return fs == s and RANK[r] == RANK[fr] + 1

    def can_stack(src_card, dest_card, dest_has_fd):
        if dest_card is None:
            return not dest_has_fd and src_card[:-1] == 'K'
        if dest_card == "fd" or not dest_card:
            return False
        sr, ss = src_card[:-1], src_card[-1]
        dr, ds = dest_card[:-1], dest_card[-1]
        if sr not in RANK or dr not in RANK:
            return False
        return RANK[sr] + 1 == RANK[dr] and (
            (is_red(ss) and is_black(ds)) or (is_black(ss) and is_red(ds))
        )

    def is_valid_sequence(seq):
        for i in range(len(seq)-1):
            r1, s1 = seq[i][:-1], seq[i][-1]
            r2, s2 = seq[i+1][:-1], seq[i+1][-1]
            if RANK[r1] != RANK[r2] + 1:
                return False
            if (is_red(s1) and is_red(s2)) or (is_black(s1) and is_black(s2)):
                return False
        return True

    def in_foundation(card):
        if card is None:
            return False
        return any(card == f for f in gs['foundation'] if f is not None)

    def lower_opposites_ready(card):
        r, s = card[:-1], card[-1]
        if r == 'A':
            return True
        prev_rank = [k for k, v in RANK.items() if v == RANK[r] - 1][0]
        if is_red(s):
            needed = [prev_rank + 's', prev_rank + 'c']
        else:
            needed = [prev_rank + 'h', prev_rank + 'd']
        return all(in_foundation(c) for c in needed)

    def apply_move(game_state, move):
        if move is None:
            return game_state

        gs = copy.deepcopy(game_state)
        cards = move['cards']
        src = move['from']
        dst = move['to']

        # -------- REMOVE FROM SOURCE --------
        if src[0] == 'tablaue':
            pile_idx = src[1]
            pile = gs['tablaue'][pile_idx]
            if len(src) == 3:
                card_idx = src[2]
                del pile[card_idx:card_idx + len(cards)]
            else:
                for _ in range(len(cards)):
                    pile.pop()
        else:
            for _ in range(len(cards)):
                gs[src[0]].pop()

        # -------- ADD TO DESTINATION --------
        zone = dst[0]
        if zone == 'tablaue':
            gs['tablaue'][dst[1]].extend(cards)
        else:
            gs[zone].extend(cards)

        return gs

    moves = []

    # ----------------------------
    # Tableau -> Foundation
    # ----------------------------
    for ti, pile in enumerate(gs['tablaue']):
        if not pile:
            continue
        start = first_faceup(pile)
        if start >= len(pile):
            continue

        card = pile[-1]
        rank_val = RANK[card[:-1]]

        for fi in range(4):
            fcard = gs['foundation'][fi]
            if can_foundation(card, fcard):
                if rank_val < 4:
                    score = 500
                else:
                    score = 200

                if lower_opposites_ready(card):
                    if rank_val < 4:
                        score += 30
                    else:
                        score += 20
                else:
                    if rank_val < 4:
                        score -= 10
                    else:
                        score -= 50

                if has_fd(pile) and len(pile) - 1 == start:
                    score += 90

                moves.append({
                    'score': score,
                    'move': {
                        'cards': [card],
                        'from': ('tablaue', ti, len(pile)-1),
                        'to': ('foundation', fi),
                        'dest_last_card': fcard
                    }
                })

    # ----------------------------
    # Tableau -> Tableau
    # ----------------------------
    for si, src in enumerate(gs['tablaue']):
        if not src:
            continue

        start = first_faceup(src)
        while start < len(src):
            seq = src[start:]
            if is_valid_sequence(seq):
                break
            start += 1
        else:
            continue

        for di, dst in enumerate(gs['tablaue']):
            if si == di:
                continue

            dest_card = dst[-1] if dst and dst[-1] != "fd" else None
            dest_has_fd = has_fd(dst)

            if can_stack(seq[0], dest_card, dest_has_fd):

                if len(seq) >= 1 and seq[0][:-1] == 'K' and dest_card is None and not has_fd(src) and len(src) >= 1:
                    continue

                if dest_card is None:
                    empty_tops = [pile[-1] for pile in gs['tablaue'] if pile and pile[-1] == 'fd']
                    if seq[0] in empty_tops:
                        continue

                if seq[0][:-1] == 'K' and dest_card is None:
                    score = 60
                    score += len(seq) * 2
                    moves.append({
                        'score': score,
                        'move': {
                            'cards': list(seq),
                            'from': ('tablaue', si, start),
                            'to': ('tablaue', di),
                            'dest_last_card': dest_card
                        }
                    })
                    continue

                score = 150

                if has_fd(src) and start == first_faceup(src):
                    score += 100
                    score += si * 5

                would_empty = (
                    not has_fd(src) and
                    start == 0 and
                    len(seq) == len(src)
                )
                if would_empty:
                    score += 80

                score += len(seq) * 2

                moves.append({
                    'score': score,
                    'move': {
                        'cards': list(seq),
                        'from': ('tablaue', si, start),
                        'to': ('tablaue', di),
                        'dest_last_card': dest_card
                    }
                })

    # ----------------------------
    # Waste -> Foundation
    # ----------------------------
    if gs['waste']:
        card = gs['waste'][0]
        rank_val = RANK[card[:-1]]

        for fi in range(4):
            fcard = gs['foundation'][fi]
            if can_foundation(card, fcard):
                if rank_val < 4:
                    score = 520
                else:
                    score = 45

                if lower_opposites_ready(card):
                    if rank_val < 4:
                        score += 30
                    else:
                        score += 20
                else:
                    if rank_val < 4:
                        score -= 10
                    else:
                        score -= 50

                moves.append({
                    'score': score,
                    'move': {
                        'cards': [card],
                        'from': ('waste', 0),
                        'to': ('foundation', fi),
                        'dest_last_card': fcard
                    }
                })

        # ----------------------------
        # Waste -> Tableau
        # ----------------------------
        for ti, dst in enumerate(gs['tablaue']):
            dest_card = dst[-1] if dst and dst[-1] != "fd" else None
            dest_has_fd = has_fd(dst)

            if can_stack(card, dest_card, dest_has_fd):
                if dest_card is None and empty_top.get(ti, None) == card:
                    continue

                score = 200

                if has_fd(dst):
                    score += 20

                moves.append({
                    'score': score,
                    'move': {
                        'cards': [card],
                        'from': ('waste', 0),
                        'to': ('tablaue', ti),
                        'dest_last_card': dest_card
                    }
                })


    for si, pile in enumerate(gs['tablaue']):
        if not pile:
            continue
        
        start = first_faceup(pile)
        if start >= len(pile):
            continue
        
        # iterate through visible cards, not just top
        for idx in range(start, len(pile) - 1):  # exclude top since not blocked
            target_card = pile[idx]

            # can this blocked card go to foundation?
            playable_to_foundation = False
            for fi in range(4):
                if can_foundation(target_card, gs['foundation'][fi]):
                    playable_to_foundation = True
                    break
                
            if not playable_to_foundation:
                continue
            
            # cards above target are blockers
            blockers = pile[idx + 1:]

            # try moving blockers somewhere else
            moved_all = True
            current_blockers = list(blockers)

            while current_blockers:
                moved = False

                # try longest valid movable suffix first
                for b_start in range(len(current_blockers)):
                    seq = current_blockers[b_start:]
                    if not is_valid_sequence(seq):
                        continue
                    
                    first_card = seq[0]

                    for di, dst in enumerate(gs['tablaue']):
                        if di == si:
                            continue
                        
                        dest_card = dst[-1] if dst and dst[-1] != "fd" else None
                        dest_has_fd = has_fd(dst)

                        if can_stack(first_card, dest_card, dest_has_fd):
                            moves.append({
                                'score': 300 + len(seq) * 10,
                                'move': {
                                    'cards': list(seq),
                                    'from': ('tablaue', si, idx + 1 + b_start),
                                    'to': ('tablaue', di),
                                    'dest_last_card': dest_card
                                }
                            })

                            # remove moved cards from temp blockers
                            current_blockers = current_blockers[:b_start]
                            moved = True
                            break
                        
                    if moved:
                        break
                    
                if not moved:
                    moved_all = False
                    break
                
            # if all blockers removable, prefer strongly
            if moved_all and blockers:
                pass

    # ----------------------------
    # Return best move
    # ----------------------------
    if not moves:
        # only append None (click stock) if moves_list is empty
        # don't add stock click if we already have real moves queued
        if not moves_list:
            moves_list.append(None)
        return moves_list

    best = max(moves, key=lambda x: x['score'])
    best_move = best['move']

    # only append None if no real moves exist yet in the list
    if best_move is None and moves_list:
        return moves_list

    moves_list.append(best_move)

    # apply move to game state for next recursion
    game_state = apply_move(game_state, best_move)

    # update empty_top — guarded safely
    to_loc = best_move.get('to')
    if (to_loc is not None
            and to_loc[0] == 'tablaue'
            and isinstance(to_loc[1], int)
            and best_move.get('dest_last_card') is None):
        empty_top[to_loc[1]] = best_move['cards'][0]

    # recurse for next move
    if depth > 1:
        moves_list = get_best_move(
            game_state, moves_list, depth - 1, empty_top=empty_top
        )

    return moves_list
#Mouse function
def execute_move(move, cards_coords,delay,stock_delay,speed=3000,flag=True, initial_flag=True):
    if move is None:
        card_coords = cards_coords['stock_0']
        print("Action: Click stock")
        pyautogui.moveTo(card_coords['cx'], card_coords['cy'])
        pyautogui.click()
        time.sleep(stock_delay + 0.2)

    # ---------------------------------------------------
    # CASE 2: Move to EMPTY FOUNDATION (Ace placement)
    # ---------------------------------------------------
    elif move['dest_last_card'] is None and move['to'][0] == 'foundation':
        card_name = move['cards'][0]

        card_coords = cards_coords[card_name]
        dest_coords = cards_coords[f"f_{move['to'][1]}"]
        cards_coords[card_name] = dest_coords
        print("Action: Move card to empty foundation")
        print("Card:", card_name, "from", card_coords)
        print("Destination:", dest_coords)
        pyautogui.moveTo(x=card_coords['cx']+10, y=card_coords['cy'])
        time.sleep(0.05)
        pyautogui.click()
        time.sleep(0.03)
        pyautogui.click()
        time.sleep(0.1)
        if flag:
            pyautogui.moveTo(x=20, y=20)

    # ---------------------------------------------------
    # CASE 2b: Move to EMPTY TABLEAU
    # ---------------------------------------------------
    elif move['dest_last_card'] is None and move['to'][0] == 'tablaue':
        card_name = move['cards'][0]

        card_coords = cards_coords[card_name]
        dest_coords = cards_coords[f"t_{move['to'][1]}"]
        dest_coords['cx'] = dest_coords['cx'] + 15
        dest_coords['cy'] = dest_coords['cy'] -30
        dist = math.hypot(card_coords['cx'] - dest_coords['cx'], card_coords['cy'] - dest_coords['cx'])
        duration = dist / speed
        cards_coords[card_name] = dest_coords
        print("Action: Move card to empty tableau")
        print("Card:", card_name, "from", card_coords)
        print("Destination:", dest_coords)
        
        pyautogui.mouseDown(x=card_coords['cx'], y=card_coords['cy'])
        pyautogui.mouseDown(x=card_coords['cx'], y=card_coords['cy'])
        pyautogui.moveTo(dest_coords['cx'], dest_coords['cy'], duration=duration)

        time.sleep(delay)
        pyautogui.mouseUp()
        time.sleep(0.1)
        if flag:
            pyautogui.moveTo(x=20, y=20)

    # ---------------------------------------------------
    # CASE 3: Move onto another card (tableau / foundation)
    # ---------------------------------------------------
    else:
        if move['to'][0] == 'foundation':
            card_name = move['cards'][0]
            card_coords = cards_coords[card_name]
            
            print("Action: Move card to empty foundation")
            print("Card:", card_name, "from", card_coords)
            if move['from'][0] == 'tablaue' and initial_flag == False:
                pyautogui.moveTo(x=card_coords['cx']+10, y=card_coords['cy'])
            else:
                pyautogui.moveTo(x=card_coords['cx']+10, y=card_coords['cy']+20)
            time.sleep(0.05)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(0.05)
            if flag:
                pyautogui.moveTo(x=20, y=20)

        else:
            
            card_name = move['cards'][0]
            card_coords = cards_coords[card_name]

            dest_card_name = move['dest_last_card']
            dest_coords = cards_coords[dest_card_name]

            dist = math.hypot(card_coords['cx'] - dest_coords['cx'], card_coords['cy'] - dest_coords['cx'])
            duration = dist / speed

            cards_coords[card_name] = cards_coords[dest_card_name]
            dest_coords['cx'] = dest_coords['cx'] + 15 
            dest_coords['cy'] = dest_coords['cy'] + 20
            print("Action: Stack card(s)")
            print("Moving:", move['cards'])
            print("From:", card_coords)
            print("Onto:", dest_card_name, "at", dest_coords)
            pyautogui.mouseDown(x=card_coords['cx'] , y=card_coords['cy'] )
            pyautogui.moveTo(dest_coords['cx'], dest_coords['cy'], duration=duration)
            time.sleep(delay)
            pyautogui.mouseUp()
            if flag:
                time.sleep(delay*0.5)
                pyautogui.moveTo(dest_coords['cx']+100, y=10)
                
            else:
                time.sleep(0.1)

            if move['from'][0] == 'waste' and flag:
                time.sleep(0.3)

    # Reset cursor
    if flag:
        time.sleep(0.2)

def generate_detection_mask(moves, game_state):
    """
    Given a list of moves from get_best_move() and current game_state,
    returns a detection_mask where 1 = re-detect, 0 = skip.
    Unions all affected regions across all moves in the list.
    """
    # start with everything as 0
    mask = {
        'foundation': [0, 0, 0, 0],
        'stock':      [0],
        'tablaue':    [0, 0, 0, 0, 0, 0, 0],
        'waste':      [0]
    }

    # normalize — accept single move or list
    if moves is None:
        moves = [None]
    elif not isinstance(moves, list):
        moves = [moves]

    for move in moves:

        # if no move, only re-detect waste
        if move is None:
            mask['waste'][0] = 1
            continue

        from_loc = move.get('from')
        to_loc   = move.get('to')

        # ---------------- MARK SOURCE ----------------
        if from_loc is not None:
            src_type = from_loc[0]

            if src_type == 'tablaue':
                idx = from_loc[1]
                mask['tablaue'][idx] = 1

            elif src_type == 'foundation':
                idx = from_loc[1]
                mask['foundation'][idx] = 1

            elif src_type == 'waste':
                mask['waste'][0] = 1
                mask['stock'][0] = 1

            elif src_type == 'stock':
                mask['stock'][0] = 1
                mask['waste'][0] = 1

        # ---------------- MARK DESTINATION ----------------
        if to_loc is not None:
            dst_type = to_loc[0]

            if dst_type == 'tablaue':
                idx = to_loc[1]
                mask['tablaue'][idx] = 1

            elif dst_type == 'foundation':
                idx = to_loc[1]
                mask['foundation'][idx] = 1

            elif dst_type == 'waste':
                mask['waste'][0] = 1

        # ---------------- SPECIAL CASE: fd reveal ----------------
        if from_loc is not None and from_loc[0] == 'tablaue':
            idx = from_loc[1]
            pile = game_state['tablaue'][idx]
            cards_moved = move.get('cards', [])
            remaining = len(pile) - len(cards_moved)
            if remaining >= 1 and pile[0] == 'fd':
                mask['tablaue'][idx] = 1  # already 1, explicit

    return mask

# Global variables
stock_line = []       # list of revealed cards in stock order
current_idx = 0       # index of currently visible waste card in stock_line
cycle_frozen = False  # True once repetition detected
total_cards = 24      # initially
pending_refresh_idx = None  # index from which next cycle will refresh cards
last_move = None
last_waste = None


def on_stock_click(detected_card):
    """
    detected_card: card currently detected in waste after stock click
    """
    global stock_line, current_idx, cycle_frozen, total_cards, pending_refresh_idx

    if detected_card is None:
        # None at end of cycle → move current index forward
        current_idx = 0
        # Start of new cycle: refresh cards after pending_refresh_idx
        if pending_refresh_idx is not None:
            for i in range(pending_refresh_idx, len(stock_line)):
                stock_line[i] = None # clear for new cycle
            pending_refresh_idx = None
        return stock_line, current_idx, cycle_frozen

    # Fill first None slot if present
    if not cycle_frozen:
        try:
            idx_none = stock_line.index(None)
            stock_line[idx_none] = detected_card
        except ValueError:
            # All filled → append if new
            if detected_card not in stock_line:
                stock_line.append(detected_card)

        # Detect repetition
        if stock_line.count(detected_card) > 1:
            cycle_frozen = True

    # Update current index
    if detected_card in stock_line:
        current_idx = stock_line.index(detected_card)

    return stock_line, current_idx, cycle_frozen


def on_waste_used(card, replacement_card=None):
    """
    card: card used from waste
    replacement_card: new card that replaces it in current cycle
    """
    global stock_line, current_idx, cycle_frozen, total_cards, pending_refresh_idx

    if card not in stock_line:
        return stock_line, current_idx, cycle_frozen

    idx = stock_line.index(card)

    # Replace used card in current cycle
    stock_line[idx] = replacement_card if replacement_card else None

    # Mark the index for future cycle refresh
    if pending_refresh_idx:
      if pending_refresh_idx < idx:
        pending_refresh_idx = idx
    else:
        pending_refresh_idx = idx  # everything after this idx will be refreshed in next cycle

    # Current index shifts to used card
    current_idx = idx

    # Cycle can unfreeze (new cards might appear)
    cycle_frozen = False

    # Update total tracked cards
    total_cards = len([c for c in stock_line if c is not None])

    return stock_line, current_idx, cycle_frozen

def update_stock_waste_state(move, waste):
    """
    move: move dict or None
    waste: current game_state['waste'] list
    """
    global last_move, last_waste, stock_line

    # --------------------------------------------------
    # 1. Resolve previous waste usage
    # --------------------------------------------------
    if last_move is not None and last_move['from'][0] == 'waste':
        replacement = waste[0] if waste else None
        on_waste_used(last_waste, replacement)
        print("waste used")

    # --------------------------------------------------
    # 2. Stock click always happens after resolution
    # --------------------------------------------------
    if waste:
        on_stock_click(waste[0])
        print("waste")
    else:
        on_stock_click(None)
        print("stock")

    # --------------------------------------------------
    # 3. Update trackers
    # --------------------------------------------------
    last_move = move
    last_waste = waste[0] if waste else None


class SolverThread(QtCore.QThread):
    update_ui = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.stop_flag = False
        self.scales = [0.95,0.975,1.0,1.025,1.05]
        self.delay = 0.3
        self.stock_delay = 0.6
        self.speed = 3000
        self.moves_per_detection = 1
        self.auto_restart = False
        self.refresh = 10
        
    def run(self):
        global templates, stock_line,current_idx,ycle_frozen,total_cards,pending_refresh_idx,last_move,last_waste
        stock_line = []       # list of revealed cards in stock order
        current_idx = 0       # index of currently visible waste card in stock_line
        cycle_frozen = False  # True once repetition detected
        total_cards = 24      # initially
        pending_refresh_idx = None  # index from which next cycle will refresh cards
        last_move = None
        last_waste = None
        none_counter = -1
        card_count = 24
        done = True
        count = 0
        move_count = 0
        move_time = 0
        detection_time = 0
        x1 = y1 = x2 = y2 = None
        move_speed = 0
        cards_coords = None
        game_state = None
        mask_count = 0
        stock_start = False
        now = 1
        full_counter = 0
        detection_mask = {
            'foundation': [1, 1, 1, 1],   # only re-detect f2 and f3
            'stock':      [0],             # skip stock
            'tablaue':    [1, 1, 1, 1, 1, 1, 1],  # only re-detect t4 and t6
            'waste':      [1]              # re-detect waste
        }


        while not self.stop_flag:
            if not self.running:
                self.msleep(50)
                continue

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                if done:
                    done = False
                    x1, y1, x2, y2 = compute_crop_roi(img, "ROI_final.json")

            
            if now == 0:
                full_start = time.time()
            else:
                count += 1
                start = time.time()
            mask_count = mask_count + 1
            
            game_state, cards_coords = detect_solitaire_cards(
                img, templates, x1, y1, x2, y2,scales = self.scales, num_workers=8,detection_mask=detection_mask,
            prev_game_state=game_state,
            prev_cards_coords=cards_coords
            )
            if now == 0:
                full_counter +=1
                full_time = time.time() - full_start
                avg_full = full_time / full_counter
                print(avg_full)
            else:
                detection_time += time.time() - start
                detection_speed = detection_time / count

            start = time.time()
            game_state = correct_tableau_order(game_state)
            print(game_state)
            if stock_start:
                update_stock_waste_state(move,game_state['waste'])

            for i in game_state['tablaue']:
                if len(i) == 1 and i[0] == 'fd':
                    del i[0]

            moves = []
            moves = get_best_move(game_state,moves,self.moves_per_detection,stock_line)
            print('move: ',moves)
            if moves is not None:
                move = moves[0]
            else:
                move =  None
            
            detection_mask = generate_detection_mask(moves, game_state)
            now = 1
            if mask_count > self.refresh:
                detection_mask = {
                'foundation': [1, 1, 1, 1],   # only re-detect f2 and f3
                'stock':      [0],             # skip stock
                'tablaue':    [1, 1, 1, 1, 1, 1, 1],  # only re-detect t4 and t6
                'waste':      [1]              # re-detect waste
                }
                mask_count = 0
                now = 0

            if not stock_start:
                stock_start = True
                last_move = move

            if (
                len(game_state["foundation"]) > 0
                or len(game_state["stock"]) > 0
                or len(game_state["waste"]) > 0
                or any(len(col) > 0 for col in game_state["tablaue"])
            ):
                # At least one pile contains something
                
                if not self.running:
                        break
                if game_state['waste']==[] and move is not None and card_count > 0:
                    time.sleep(0.1)
                    card_coords = cards_coords['stock_0']
                    print("Action: Click stock")
                    pyautogui.moveTo(card_coords['cx'], card_coords['cy'])
                    pyautogui.click()
                    time.sleep(0.1)
                    detection_mask['waste'] = [1]
                print(detection_mask)
                for i, move in enumerate(moves):
                    print(move)
                    if i == len(moves) -1:
                        execute_move(move, cards_coords, self.delay,self.stock_delay, self.speed,flag=True)
                    elif i == 0:
                        execute_move(move, cards_coords, self.delay,self.stock_delay, self.speed,flag=False,initial_flag = True)
                    else:
                        execute_move(move, cards_coords, self.delay,self.stock_delay, self.speed,flag=False)

                    if move == None:
                        none_counter += 1
                    else:
                        if move['from'][0] == 'waste':
                            card_count -= 1
                        none_counter = 0

                    none_con = card_count // 3
                    if card_count % 3 != 0:
                        none_con +=1

                    if self.auto_restart and none_counter >= none_con*3:
                        pyautogui.moveTo(x=1090, y=945)
                        pyautogui.click()
                        time.sleep(1)
                        pyautogui.moveTo(x=933, y=621)
                        pyautogui.click()
                        break

                    move_count += 1
                    move_time += time.time() - start
                    move_speed = move_time / max(1, move_count)

            else:
                pass
                    
            self.update_ui.emit({
                "detection_speed": detection_speed,
                "move_speed": move_speed,
                "game_state": game_state,
            })

    def stop(self):
        self.stop_flag = True
        self.running = False

def format_game_state(gs):
    """
    Return a fixed-slot pretty-print of solitaire game state.
    Foundations: 4 slots
    Stock/Waste: 1 slot each, top-right
    Tableau: 7 columns below, aligned with foundations
    """
    from copy import deepcopy
    foundations = deepcopy(gs['foundation'])
    tableau = deepcopy(gs['tablaue'])
    stock = deepcopy(gs['stock'])
    waste = deepcopy(gs['waste'])

    # Map suits to symbols
    suit_symbols = {'h':'♥', 'd':'♦', 'c':'♠', 's':'♣'}

    def card_str(card):
        if card == "fd":
            return "[fd]"
        if card is None:
            return "[  ]"
        rank, suit = card[:-1], card[-1]
        return f"[{rank}{suit_symbols.get(suit,suit)}]"

    # --- Fixed slots for foundations ---
    while len(foundations) < 4:
        foundations.append(None)
    foundation_row = "  ".join(f"{card_str(f):^5}" for f in foundations)

    # --- Stock / Waste ---
    stock_str = card_str(stock[0]) if stock else "[  ]"
    waste_str = card_str(waste[0]) if waste else "[  ]"
    top_row = f"{foundation_row}{' '*10}Stock: {stock_str}   Waste: {waste_str}"

    # --- Fixed slots for tableau ---
    # Ensure 7 columns
    while len(tableau) < 7:
        tableau.append([])

    # Find max column height
    max_height = max(len(col) for col in tableau)

    tableau_rows = []
    for row_idx in range(max_height):
        row_str = ""
        for col in tableau:
            if row_idx < len(col):
                row_str += f"{card_str(col[row_idx]):^5}"
                row_str += "  "
            else:
                row_str += "[    ]".center(5)  # Empty slot
                row_str += "   "
              # space between columns
        tableau_rows.append(row_str.rstrip())

    return top_row + "\n" + "="*40 + "\n" + "\n".join(tableau_rows)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solitaire Bot")
        self.setFixedSize(420, 1000)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.setStyleSheet("""
            QWidget { background:#0f0f0f; color:#e5e5e5; font-size:12px; }
            QPushButton {
                background:#1b1b1b; border:1px solid #2f2f2f;
                padding:6px 10px; border-radius:8px;
            }
            QPushButton:hover { background:#262626; }
            QSlider::groove:horizontal { height:6px; background:#2a2a2a; border-radius:3px; }
            QSlider::handle:horizontal {
                width:14px; background:#4cafef; margin:-4px 0; border-radius:7px;
            }
            QTextEdit, QLineEdit {
                background:#0e0e0e; border:1px solid #2f2f2f;
                padding:4px; font-family:monospace;
            }
            QGroupBox {
                border:1px solid #2f2f2f;
                border-radius:8px;
                margin-top:8px;
                padding:6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left:8px;
                padding:0 4px;
                color:#9ccc65;
            }
        """)

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(10)
        top = QtWidgets.QHBoxLayout()
        self.top_btn = QtWidgets.QPushButton("On Top: ON")
        self.top_btn.setCheckable(True)
        self.top_btn.setChecked(True)
        self.top_btn.clicked.connect(self.toggle_on_top)

        top.addWidget(self.top_btn)



        # --- Top bar ---
        
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.guide_btn = QtWidgets.QPushButton("Guide")
        self.restart_btn = QtWidgets.QPushButton("Auto-Quit: OFF")
        self.restart_btn.setCheckable(True)
        self.restart_btn.setChecked(False)
        self.restart_btn.clicked.connect(self.toggle_restart)
        top.addWidget(self.restart_btn)

        self.status_dot = QtWidgets.QLabel("●")
        self.status_dot.setStyleSheet("color:#555; font-size:16px;")

        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addStretch()
        top.addWidget(self.status_dot)
        top.addWidget(self.guide_btn)
        root.addLayout(top)

        # --- Groups ---
        perf_box, perf = self.make_group("Performance")
        perf.addLayout(self.make_slider_block("Speed", 0, 100, 100, "0", "100", lambda v: f"{v}%"))
        perf.addLayout(self.make_slider_block("Moves / Detection", 1, 5, 5, "1", "5", str))
        root.addWidget(perf_box)

        timing_box, timing = self.make_group("Timing")
        timing.addLayout(self.make_slider_block("Delay (s)", 0, 20, 6, "0", "1", lambda v: f"{v*0.05:.2f}"))
        timing.addLayout(self.make_slider_block("Stock Delay (s)", 0, 20, 3, "0", "2", lambda v: f"{v*0.1:.2f}"))
        root.addWidget(timing_box)

        vision_box, vision = self.make_group("Vision")
        vision.addLayout(self.make_slider_block("Tolerance (%)", 0, 10, 5, "-5", "5", lambda v: f"±{v-5}%"))
        vision.addLayout(self.make_slider_block("Intervals", 1, 3, 1, "1", "5", lambda v: f"{v*2-1}"))
        vision.addLayout(self.make_slider_block("Refresh", 1, 30, 20, "1", "30", lambda v: f"{v}"))
        root.addWidget(vision_box)

        # --- Stats ---
        stats = QtWidgets.QHBoxLayout()
        self.det_box = QtWidgets.QLineEdit("0.000 s")
        self.move_box = QtWidgets.QLineEdit("0.000 s")
        self.det_box.setReadOnly(True)
        self.move_box.setReadOnly(True)
        stats.addWidget(QtWidgets.QLabel("⏱ Detect"))
        stats.addWidget(self.det_box)
        stats.addWidget(QtWidgets.QLabel("🖱 Move"))
        stats.addWidget(self.move_box)
        root.addLayout(stats)

        self.last_label = QtWidgets.QLabel("Last Move: —")
        root.addWidget(self.last_label)

        # --- State ---
        self.state_toggle = QtWidgets.QPushButton("Show State")
        self.state_toggle.setCheckable(True)
        root.addWidget(self.state_toggle)

        self.state_box = QtWidgets.QTextEdit()
        self.state_box.setReadOnly(True)
        self.state_box.setVisible(False)
        root.addWidget(self.state_box)

        self.state_toggle.toggled.connect(self.state_box.setVisible)

        # --- Guide ---
        self.guide_box = QtWidgets.QTextEdit()
        self.guide_box.setReadOnly(True)
        self.guide_box.setVisible(False)
        self.guide_box.setFixedHeight(300)
        self.state_box.setMinimumHeight(200)
        self.state_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.guide_box.setText(
            "Speed:\nControls mouse velocity.\n\n"
            "Delay / Stock Delay:\nAllow UI to settle between moves.\n\n"
            "Moves / Detection:\nHow many moves are executed per scan.\n\n"
            "Tolerance / Intervals:\nTemplate scale search range. Interval decides how many different\nscales model runs in one cycle, more interval slower detections but more accurate\n\n"
            "Refresher: Runs on all tableau after this number of moves to \n correct previous misdetections"
        )
        root.addWidget(self.guide_box)

        self.guide_btn.clicked.connect(
            lambda: self.guide_box.setVisible(not self.guide_box.isVisible())
        )

        # --- Worker ---
        self.worker = SolverThread()
        self.worker.update_ui.connect(self.on_update)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        keyboard.add_hotkey('s', self._safe_start)
        keyboard.add_hotkey('q', self._safe_stop)

        # --- Glow ---
        self.add_glow(self.start_btn, "#4cafef", 22)
        self.add_glow(self.stop_btn,  "#ef4444", 22)
        self.add_glow(self.guide_btn, "#a78bfa", 18)
        self.add_glow(self.restart_btn, "#9ccc65", 18)

        for s in [
            self.speed_slider, self.delay_slider, self.stock_slider,
            self.moves_slider, self.tolerance_slider, self.intervals_slider
        ]:
            self.add_glow(s, "#4cafef", 12)

    # ---------- Helpers ----------
    def _safe_start(self):
        QtCore.QMetaObject.invokeMethod(self, "start", QtCore.Qt.QueuedConnection)

    def _safe_stop(self):
        QtCore.QMetaObject.invokeMethod(self, "stop", QtCore.Qt.QueuedConnection)
        
    def make_group(self, title):
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        return box, layout

    def make_slider_block(self, title, mn, mx, val, ltxt, rtxt, fmt):
        box = QtWidgets.QVBoxLayout()

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel(title))
        header.addStretch()
        value_box = QtWidgets.QLineEdit()
        value_box.setFixedWidth(60)
        value_box.setReadOnly(True)
        header.addWidget(value_box)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(val)

        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(QtWidgets.QLabel(ltxt))
        footer.addStretch()
        footer.addWidget(QtWidgets.QLabel(rtxt))

        def update(v):
            value_box.setText(fmt(v))

        slider.valueChanged.connect(update)
        update(val)

        box.addLayout(header)
        box.addWidget(slider)
        box.addLayout(footer)

        if title == "Speed": self.speed_slider = slider
        elif title == "Delay (s)": self.delay_slider = slider
        elif title == "Stock Delay (s)": self.stock_slider = slider
        elif title == "Moves / Detection": self.moves_slider = slider
        elif title == "Tolerance (%)": self.tolerance_slider = slider
        elif title == "Intervals": self.intervals_slider = slider
        elif title == "Refresh": self.refresh_slider = slider

        return box

    def toggle_on_top(self, checked):
        flags = self.windowFlags()

        if checked:
            flags |= QtCore.Qt.WindowStaysOnTopHint
            self.top_btn.setText("On Top: ON")
        else:
            flags &= ~QtCore.Qt.WindowStaysOnTopHint
            self.top_btn.setText("On Top: OFF")

        self.setWindowFlags(flags)
        self.show()   # required to apply change


    def add_glow(self, widget, color, blur):
        e = QtWidgets.QGraphicsDropShadowEffect(widget)
        e.setBlurRadius(blur)
        e.setOffset(0, 0)
        e.setColor(QtGui.QColor(color))
        widget.setGraphicsEffect(e)

    # ---------- Control ----------

    @QtCore.pyqtSlot()
    def start(self):
        if self.worker.isRunning():
            # Thread is alive but paused — just resume it
            self.worker.running = True
            self.status_dot.setStyleSheet("color:#4caf50; font-size:16px;")
            return

        # Thread is fully stopped — create a fresh one
        self.worker = SolverThread()
        self.worker.update_ui.connect(self.on_update)
        self.worker.running = True
        self.worker.start()
        self.status_dot.setStyleSheet("color:#4caf50; font-size:16px;")

    @QtCore.pyqtSlot()
    def stop(self):
        self.worker.stop()
        self.status_dot.setStyleSheet("color:#f44336; font-size:16px;")
    
    def toggle_restart(self, checked):
        if checked:
            self.restart_btn.setText("Auto-Quit: ON")
            self.restart_btn.setStyleSheet("background:#1a3a1a; border:1px solid #4caf50;")
        else:
            self.restart_btn.setText("Auto-Quit: OFF")
            self.restart_btn.setStyleSheet("")
        self.worker.auto_restart = checked

    def on_update(self, data):
        self.det_box.setText(f"{data['detection_speed']:.3f} s")
        self.move_box.setText(f"{data['move_speed']:.3f} s")
        pretty_state = format_game_state(data["game_state"])
        self.state_box.setText(pretty_state)

        sp = self.speed_slider.value()
        self.worker.speed = 2500 + (sp / 100) * 7000
        self.worker.delay = self.delay_slider.value() * 0.05
        self.worker.stock_delay = self.stock_slider.value() * 0.1
        self.worker.moves_per_detection = self.moves_slider.value()
        self.refresh = self.refresh_slider.value()
        
        tol = self.tolerance_slider.value()
        if tol < 6:
            tol = 5 - tol
        else:
            tol = tol - 5
        tol = tol / 100
        n = self.intervals_slider.value() * 2 -1
        
        if n == 1:
            tol = 1.0 + tol
            scales = [tol]
        else:
            if tol < 0:
                tol = tol * -1
            step = tol / ((n-1)//2)  # step based on half intervals
            mid = (n-1)//2
            scales = [1 + (i - mid) * step for i in range(n)]
        
        self.worker.scales = scales


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())










