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
import onnxruntime as ort

print("Parent Folder: ", os.getcwd())

# ---------------- LOAD YOLO MODEL ----------------
_ort_opts = ort.SessionOptions()
_ort_opts.intra_op_num_threads = 4
_ort_opts.inter_op_num_threads = 2
_ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "best.onnx",
    sess_options=_ort_opts,
    providers=["CPUExecutionProvider"]
)
YOLO_SIZE = 640
class_names = ['10', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'J', 'K', 'Q', 'c', 'd', 'fd', 'h', 's']



# Warmup pass
_dummy = np.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), dtype=np.float32)
session.run(None, {session.get_inputs()[0].name: _dummy})

# ---------------- LOAD REGIONS ONCE ----------------
with open("regions.json") as _f:
    _regions_raw = json.load(_f)

_denorm_regions_global = {
    name: (
        int(r["x1"] * YOLO_SIZE), int(r["y1"] * YOLO_SIZE),
        int(r["x2"] * YOLO_SIZE), int(r["y2"] * YOLO_SIZE),
    )
    for name, r in _regions_raw.items()
}


# ------------------------------------------------------------------ #
#  HELPERS (unchanged)                                                 #
# ------------------------------------------------------------------ #

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

        if face_up:
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

    min_x = min_y = 1.0
    max_x = max_y = 0.0

    for r in regions.values():
        min_x = min(min_x, r["x1"])
        min_y = min(min_y, r["y1"])
        max_x = max(max_x, r["x2"])
        max_y = max(max_y, r["y2"])

    x1 = max(0, int(min_x * w)) - 10
    y1 = max(0, int(min_y * h))
    x2 = min(w, int(max_x * w)) + 10
    y2 = min(h, int(max_y * h))

    return x1, y1, x2, y2


# ------------------------------------------------------------------ #
#  DETECTION  —  YOLO replaces template matching                       #
# ------------------------------------------------------------------ #

def detect_solitaire_cards(img, session, x1, y1, x2, y2,
                           conf_threshold=0.35):
    """
    Drop-in replacement for the old template-matching detect_solitaire_cards().
    Signature kept compatible: positional args img / (session instead of templates) /
    x1,y1,x2,y2 are the same; keyword args match what SolverThread passes.
    """
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    SUITS = ['c', 'd', 'h', 's']

    full_h, full_w = img.shape[:2]
    crop_x1, crop_y1 = max(0, x1), max(0, y1)
    crop_x2, crop_y2 = min(full_w, x2), min(full_h, y2)
    img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    #img_crop = cv2.imread('crop1.png')
    #cv2.imwrite('yolo_crop.png',img_crop)
    # -------- YOLO PREPROCESS --------
    img_yolo = cv2.resize(img_crop, (YOLO_SIZE, YOLO_SIZE))
    img_yolo = cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB)
    blob = img_yolo.astype(np.float32) / 255.0
    blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

    # -------- INFERENCE --------
    input_name  = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: blob})[0][0]   # (N, 6)

    all_detections = []
    for pred in predictions:
        bx1, by1, bx2, by2, conf, class_id = pred
        if conf < conf_threshold:
            continue
        cid = int(class_id)
        if cid < len(class_names):
            all_detections.append((bx1, by1, bx2, by2, class_names[cid]))



    denorm_regions = _denorm_regions_global

    # -------- ASSIGN DETECTIONS TO REGIONS --------
    def assign_region(cx, cy):
        best, best_area = None, float('inf')
        for rname, (rx1, ry1, rx2, ry2) in denorm_regions.items():
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                area = (rx2 - rx1) * (ry2 - ry1)
                if area < best_area:
                    best_area = area
                    best = rname
        return best

    region_detections = {rn: [] for rn in denorm_regions}
    for det in all_detections:
        bx1, by1, bx2, by2, cls_name = det
        rname = assign_region((bx1 + bx2) / 2, (by1 + by2) / 2)
        if rname:
            region_detections[rname].append(det)

    # -------- INITIALISE STATE --------
    cards_coords = {}
    game_state = {
        "foundation": [],
        "stock": [],
        "tablaue": [[] for _ in range(7)],
        "waste": []
    }

    

    # -------- TRACK fd COLUMNS --------
    fd_tableau = {
        rn for rn, dets in region_detections.items()
        if rn.startswith("t") and any(d[4] == "fd" for d in dets)
    }

    # -------- RANK + SUIT PAIRING --------
    scale_x = (crop_x2 - crop_x1) / YOLO_SIZE
    scale_y = (crop_y2 - crop_y1) / YOLO_SIZE

    for region_name, dets in region_detections.items():


        boxes_ranks = [(bx1, by1, bx2, by2, n) for bx1, by1, bx2, by2, n in dets if n in RANKS]
        boxes_suits = [(bx1, by1, bx2, by2, n) for bx1, by1, bx2, by2, n in dets if n in SUITS]

        added_cards = set()
        for rx1b, ry1b, rx2b, ry2b, rname in boxes_ranks:
            rcx = (rx1b + rx2b) / 2
            rcy = (ry1b + ry2b) / 2
            best_suit, min_dx = None, float('inf')

            for sx1b, sy1b, sx2b, sy2b, sname in boxes_suits:
                scx = (sx1b + sx2b) / 2
                scy = (sy1b + sy2b) / 2
                dx  = scx - rcx
                if dx > 0 and abs(scy - rcy) <= 15 and dx < min_dx:
                    min_dx    = dx
                    best_suit = sname

            if best_suit:
                card_name = rname + best_suit
                if card_name not in added_cards:
                    added_cards.add(card_name)
                    full_cx = int(rcx * scale_x + crop_x1)
                    full_cy = int(rcy * scale_y + crop_y1)
                    cards_coords[card_name] = {"cx": full_cx, "cy": full_cy}

                    if region_name.startswith("f"):
                        game_state["foundation"].append(card_name)
                    elif region_name.startswith("t"):
                        game_state["tablaue"][int(region_name[1:])].append(card_name)
                    elif region_name == "stock":
                        game_state["stock"].append(card_name)
                    elif region_name == "waste":
                        game_state["waste"].append(card_name)

    # -------- REGION SLOT CENTRES --------
    for i in range(4):
        rx1b, ry1b, rx2b, ry2b = denorm_regions[f"f{i}"]
        cards_coords[f"f_{i}"] = {
            "cx": int((rx1b + rx2b) / 2 * scale_x + crop_x1),
            "cy": int((ry1b + ry2b) / 2 * scale_y + crop_y1),
        }
    for i in range(7):
        rx1b, ry1b, rx2b, ry2b = denorm_regions[f"t{i}"]
        cards_coords[f"t_{i}"] = {
            "cx": int((rx1b + rx2b) / 2 * scale_x + crop_x1),
            "cy": int((ry1b + 0.3 * (ry2b - ry1b)) * scale_y + crop_y1),
        }
    rx1b, ry1b, rx2b, ry2b = denorm_regions["stock"]
    cards_coords["stock_0"] = {
        "cx": int((rx1b + rx2b) / 2 * scale_x + crop_x1),
        "cy": int((ry1b + ry2b) / 2 * scale_y + crop_y1),
    }

    # -------- REVERSE TABLEAUX --------
    for i in range(7):
        game_state["tablaue"][i].reverse()
        if f"t{i}" in fd_tableau:
                game_state["tablaue"][i].insert(0, "fd")


    return game_state, cards_coords


# ------------------------------------------------------------------ #
#  SOLVER  (unchanged)                                                 #
# ------------------------------------------------------------------ #

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

# ------------------------------------------------------------------ #
#  MOUSE EXECUTION  (unchanged)                                        #
# ------------------------------------------------------------------ #

def execute_move(move, cards_coords, delay, stock_delay, speed=3000, flag=True, initial_flag=True):
    if move is None:
        card_coords = cards_coords['stock_0']
        print("Action: Click stock")
        pyautogui.moveTo(card_coords['cx'], card_coords['cy'])
        pyautogui.click()
        time.sleep(stock_delay + 0.2)

    elif move['dest_last_card'] is None and move['to'][0] == 'foundation':
        card_name   = move['cards'][0]
        card_coords = cards_coords[card_name]
        dest_coords = cards_coords[f"f_{move['to'][1]}"]
        cards_coords[card_name] = dest_coords
        print("Action: Move card to empty foundation")
        print("Card:", card_name, "from", card_coords)
        print("Destination:", dest_coords)
        pyautogui.moveTo(x=card_coords['cx'] + 10, y=card_coords['cy'])
        time.sleep(0.05)
        pyautogui.click()
        time.sleep(0.03)
        pyautogui.click()
        time.sleep(0.1)
        if flag:
            pyautogui.moveTo(x=20, y=20)

    elif move['dest_last_card'] is None and move['to'][0] == 'tablaue':
        card_name   = move['cards'][0]
        card_coords = cards_coords[card_name]
        dest_coords = cards_coords[f"t_{move['to'][1]}"]
        dest_coords['cx'] = dest_coords['cx'] + 15
        dest_coords['cy'] = dest_coords['cy'] - 30
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

    else:
        if move['to'][0] == 'foundation':
            card_name   = move['cards'][0]
            card_coords = cards_coords[card_name]
            print("Action: Move card to foundation")
            print("Card:", card_name, "from", card_coords)
            if move['from'][0] == 'tablaue' and initial_flag is False:
                pyautogui.moveTo(x=card_coords['cx'] + 10, y=card_coords['cy'])
            else:
                pyautogui.moveTo(x=card_coords['cx'] + 10, y=card_coords['cy'] + 20)
            time.sleep(0.05)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(0.05)
            if flag:
                pyautogui.moveTo(x=20, y=20)
        else:
            card_name      = move['cards'][0]
            card_coords    = cards_coords[card_name]
            dest_card_name = move['dest_last_card']
            dest_coords    = cards_coords[dest_card_name]
            dist = math.hypot(card_coords['cx'] - dest_coords['cx'], card_coords['cy'] - dest_coords['cx'])
            duration = dist / speed
            cards_coords[card_name] = cards_coords[dest_card_name]
            dest_coords['cx'] = dest_coords['cx'] + 15
            dest_coords['cy'] = dest_coords['cy'] + 20
            print("Action: Stack card(s)")
            print("Moving:", move['cards'])
            print("From:", card_coords)
            print("Onto:", dest_card_name, "at", dest_coords)
            pyautogui.mouseDown(x=card_coords['cx'], y=card_coords['cy'])
            pyautogui.moveTo(dest_coords['cx'], dest_coords['cy'], duration=duration)
            time.sleep(delay)
            pyautogui.mouseUp()
            if flag:
                time.sleep(delay * 0.5)
                pyautogui.moveTo(dest_coords['cx'] + 100, y=10)
            else:
                time.sleep(0.1)
            if move['from'][0] == 'waste' and flag:
                time.sleep(0.3)

    if flag:
        time.sleep(0.2)


# ------------------------------------------------------------------ #
#  DETECTION MASK  (unchanged)                                         #
# ------------------------------------------------------------------ #

def generate_detection_mask(moves, game_state):
    mask = {
        'foundation': [0, 0, 0, 0],
        'stock':      [0],
        'tablaue':    [0, 0, 0, 0, 0, 0, 0],
        'waste':      [0]
    }

    if moves is None:
        moves = [None]
    elif not isinstance(moves, list):
        moves = [moves]

    for move in moves:
        if move is None:
            mask['waste'][0] = 1
            continue

        from_loc = move.get('from')
        to_loc   = move.get('to')

        if from_loc is not None:
            src_type = from_loc[0]
            if src_type == 'tablaue':
                mask['tablaue'][from_loc[1]] = 1
            elif src_type == 'foundation':
                mask['foundation'][from_loc[1]] = 1
            elif src_type == 'waste':
                mask['waste'][0] = 1
                mask['stock'][0] = 1
            elif src_type == 'stock':
                mask['stock'][0] = 1
                mask['waste'][0] = 1

        if to_loc is not None:
            dst_type = to_loc[0]
            if dst_type == 'tablaue':
                mask['tablaue'][to_loc[1]] = 1
            elif dst_type == 'foundation':
                mask['foundation'][to_loc[1]] = 1
            elif dst_type == 'waste':
                mask['waste'][0] = 1

        if from_loc is not None and from_loc[0] == 'tablaue':
            idx   = from_loc[1]
            pile  = game_state['tablaue'][idx]
            remaining = len(pile) - len(move.get('cards', []))
            if remaining >= 1 and pile[0] == 'fd':
                mask['tablaue'][idx] = 1

    return mask


# ------------------------------------------------------------------ #
#  STOCK / WASTE TRACKING  (unchanged)                                 #
# ------------------------------------------------------------------ #

stock_line         = []
current_idx        = 0
cycle_frozen       = False
total_cards        = 24
pending_refresh_idx = None
last_move          = None
last_waste         = None


def on_stock_click(detected_card):
    global stock_line, current_idx, cycle_frozen, total_cards, pending_refresh_idx

    if detected_card is None:
        current_idx = 0
        if pending_refresh_idx is not None:
            for i in range(pending_refresh_idx, len(stock_line)):
                stock_line[i] = None
            pending_refresh_idx = None
        return stock_line, current_idx, cycle_frozen

    if not cycle_frozen:
        try:
            idx_none = stock_line.index(None)
            stock_line[idx_none] = detected_card
        except ValueError:
            if detected_card not in stock_line:
                stock_line.append(detected_card)
        if stock_line.count(detected_card) > 1:
            cycle_frozen = True

    if detected_card in stock_line:
        current_idx = stock_line.index(detected_card)

    return stock_line, current_idx, cycle_frozen


def on_waste_used(card, replacement_card=None):
    global stock_line, current_idx, cycle_frozen, total_cards, pending_refresh_idx

    if card not in stock_line:
        return stock_line, current_idx, cycle_frozen

    idx = stock_line.index(card)
    stock_line[idx] = replacement_card if replacement_card else None

    if pending_refresh_idx:
        if pending_refresh_idx < idx:
            pending_refresh_idx = idx
    else:
        pending_refresh_idx = idx

    current_idx  = idx
    cycle_frozen = False
    total_cards  = len([c for c in stock_line if c is not None])

    return stock_line, current_idx, cycle_frozen


def update_stock_waste_state(move, waste):
    global last_move, last_waste, stock_line

    if last_move is not None and last_move['from'][0] == 'waste':
        replacement = waste[0] if waste else None
        on_waste_used(last_waste, replacement)
        print("waste used")

    if waste:
        on_stock_click(waste[0])
        print("waste")
    else:
        on_stock_click(None)
        print("stock")

    last_move  = move
    last_waste = waste[0] if waste else None


# ------------------------------------------------------------------ #
#  SOLVER THREAD                                                       #
# ------------------------------------------------------------------ #

class SolverThread(QtCore.QThread):
    update_ui = QtCore.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running           = False
        self.stop_flag         = False
        self.conf_threshold    = 0.35   # replaces template scales/tolerance
        self.delay             = 0.3
        self.stock_delay       = 0.6
        self.speed             = 3000
        self.moves_per_detection = 1
        self.auto_restart      = False

    def run(self):
        global stock_line, current_idx, cycle_frozen, total_cards
        global pending_refresh_idx, last_move, last_waste

        # reset globals
        stock_line          = []
        current_idx         = 0
        cycle_frozen        = False
        total_cards         = 24
        pending_refresh_idx = None
        last_move           = None
        last_waste          = None

        none_counter  = -1
        card_count    = 24
        done          = True
        count         = 0
        move_count    = 0
        move_time     = 0
        detection_time = 0
        x1 = y1 = x2 = y2 = None
        move_speed    = 0
        cards_coords  = None
        game_state    = None
        mask_count    = 0
        stock_start   = False

        while not self.stop_flag:
            if not self.running:
                self.msleep(50)
                continue

            with mss.mss() as sct:
                monitor    = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img        = np.array(screenshot)
                img        = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if done:
                done = False
                print('done')
                x1, y1, x2, y2 = compute_crop_roi(img, "ROI_final.json")

            count      += 1
            mask_count += 1
            start       = time.time()

            # ---- YOLO detection (session passed instead of templates) ----
            game_state, cards_coords = detect_solitaire_cards(
                img, session, x1, y1, x2, y2,
                conf_threshold=self.conf_threshold
            )

            detection_time  += time.time() - start
            detection_speed  = detection_time / count

            start       = time.time()
            game_state  = correct_tableau_order(game_state)
            print(game_state)

            if stock_start:
                update_stock_waste_state(move, game_state['waste'])

            for i in game_state['tablaue']:
                if len(i) == 1 and i[0] == 'fd':
                    del i[0]

            moves = []
            moves = get_best_move(game_state, moves, self.moves_per_detection, stock_line)
            print('move: ', moves)

            move = moves[0] if moves else None


            if not stock_start:
                stock_start = True
                last_move   = move

            if (
                len(game_state["foundation"]) > 0
                or len(game_state["stock"]) > 0
                or len(game_state["waste"]) > 0
                or any(len(col) > 0 for col in game_state["tablaue"])
            ):
                if not self.running:
                    break

                if game_state['waste'] == [] and move is not None and card_count > 0:
                    time.sleep(0.1)
                    card_coords = cards_coords['stock_0']
                    print("Action: Click stock")
                    pyautogui.moveTo(card_coords['cx'], card_coords['cy'])
                    pyautogui.click()
                    time.sleep(0.1)

                for i, move in enumerate(moves):
                    print(move)
                    if i == len(moves) - 1:
                        execute_move(move, cards_coords, self.delay, self.stock_delay, self.speed, flag=True)
                    elif i == 0:
                        execute_move(move, cards_coords, self.delay, self.stock_delay, self.speed, flag=False, initial_flag=True)
                    else:
                        execute_move(move, cards_coords, self.delay, self.stock_delay, self.speed, flag=False)

                    if move is None:
                        none_counter += 1
                    else:
                        if move['from'][0] == 'waste':
                            card_count -= 1
                        none_counter = 0

                    none_con = card_count // 3
                    if card_count % 3 != 0:
                        none_con += 1

                    if self.auto_restart and none_counter >= none_con * 3:
                        pyautogui.moveTo(x=1090, y=945)
                        pyautogui.click()
                        time.sleep(1)
                        pyautogui.moveTo(x=933, y=621)
                        pyautogui.click()
                        break

                    move_count += 1
                    move_time  += time.time() - start
                    move_speed  = move_time / max(1, move_count)

            self.update_ui.emit({
                "detection_speed": detection_speed,
                "move_speed":      move_speed,
                "game_state":      game_state,
            })

    def stop(self):
        self.stop_flag = True
        self.running   = False


# ------------------------------------------------------------------ #
#  UI HELPERS  (unchanged)                                             #
# ------------------------------------------------------------------ #

def format_game_state(gs):
    from copy import deepcopy
    foundations = deepcopy(gs['foundation'])
    tableau     = deepcopy(gs['tablaue'])
    stock       = deepcopy(gs['stock'])
    waste       = deepcopy(gs['waste'])

    suit_symbols = {'h': '♥', 'd': '♦', 'c': '♠', 's': '♣'}

    def card_str(card):
        if card == "fd":    return "[fd]"
        if card is None:    return "[  ]"
        rank, suit = card[:-1], card[-1]
        return f"[{rank}{suit_symbols.get(suit, suit)}]"

    while len(foundations) < 4:
        foundations.append(None)
    foundation_row = "  ".join(f"{card_str(f):^5}" for f in foundations)

    stock_str = card_str(stock[0]) if stock else "[  ]"
    waste_str = card_str(waste[0]) if waste else "[  ]"
    top_row   = f"{foundation_row}{' '*10}Stock: {stock_str}   Waste: {waste_str}"

    while len(tableau) < 7:
        tableau.append([])

    max_height   = max(len(col) for col in tableau)
    tableau_rows = []
    for row_idx in range(max_height):
        row_str = ""
        for col in tableau:
            if row_idx < len(col):
                row_str += f"{card_str(col[row_idx]):^5}  "
            else:
                row_str += "[    ]".center(5) + "   "
        tableau_rows.append(row_str.rstrip())

    return top_row + "\n" + "=" * 40 + "\n" + "\n".join(tableau_rows)


# ------------------------------------------------------------------ #
#  MAIN WINDOW                                                         #
# ------------------------------------------------------------------ #

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solitaire Bot")
        self.setFixedSize(420, 800)
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

        # --- Top bar ---
        top = QtWidgets.QHBoxLayout()
        self.top_btn = QtWidgets.QPushButton("On Top: ON")
        self.top_btn.setCheckable(True)
        self.top_btn.setChecked(True)
        self.top_btn.clicked.connect(self.toggle_on_top)

        top.addWidget(self.top_btn)
        self.start_btn   = QtWidgets.QPushButton("Start")
        self.stop_btn    = QtWidgets.QPushButton("Stop")
        self.guide_btn   = QtWidgets.QPushButton("Guide")
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
        # Confidence threshold replaces Tolerance/Intervals from template days
        vision.addLayout(self.make_slider_block("Confidence (%)", 10, 95, 35, "10", "95", lambda v: f"{v}%"))
        root.addWidget(vision_box)

        # --- Stats ---
        stats = QtWidgets.QHBoxLayout()
        self.det_box  = QtWidgets.QLineEdit("0.000 s")
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
        self.state_box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.guide_box.setText(
            "Speed:\nControls mouse velocity.\n\n"
            "Delay / Stock Delay:\nAllow UI to settle between moves.\n\n"
            "Moves / Detection:\nHow many moves are executed per scan.\n\n"
            "Confidence (%):\nMinimum YOLO detection confidence.\n"
            "Lower = more detections (may add noise).\n"
            "Higher = fewer but more certain detections."
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
        self.add_glow(self.start_btn,   "#4cafef", 22)
        self.add_glow(self.stop_btn,    "#ef4444", 22)
        self.add_glow(self.guide_btn,   "#a78bfa", 18)
        self.add_glow(self.restart_btn, "#9ccc65", 18)

        for s in [self.speed_slider, self.delay_slider,
                  self.stock_slider, self.moves_slider, self.conf_slider]:
            self.add_glow(s, "#4cafef", 12)

    # ---------- Helpers ----------
    def _safe_start(self):
        QtCore.QMetaObject.invokeMethod(self, "start", QtCore.Qt.QueuedConnection)

    def _safe_stop(self):
        QtCore.QMetaObject.invokeMethod(self, "stop", QtCore.Qt.QueuedConnection)
    def make_group(self, title):
        box    = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        return box, layout

    def make_slider_block(self, title, mn, mx, val, ltxt, rtxt, fmt):
        box    = QtWidgets.QVBoxLayout()
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

        if title == "Speed":               self.speed_slider = slider
        elif title == "Delay (s)":         self.delay_slider = slider
        elif title == "Stock Delay (s)":   self.stock_slider = slider
        elif title == "Moves / Detection": self.moves_slider = slider
        elif title == "Confidence (%)":    self.conf_slider  = slider

        return box

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
            self.worker.running = True
            self.status_dot.setStyleSheet("color:#4caf50; font-size:16px;")
            return
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
        self.state_box.setText(format_game_state(data["game_state"]))

        self.worker.speed             = 2500 + (self.speed_slider.value() / 100) * 12000
        self.worker.delay             = self.delay_slider.value() * 0.05
        self.worker.stock_delay       = self.stock_slider.value() * 0.1
        self.worker.moves_per_detection = self.moves_slider.value()
        self.worker.conf_threshold    = self.conf_slider.value() / 100.0
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


# ------------------------------------------------------------------ #
#  ENTRY POINT                                                         #
# ------------------------------------------------------------------ #

app = QtWidgets.QApplication(sys.argv)
w   = MainWindow()
w.show()
sys.exit(app.exec_())
