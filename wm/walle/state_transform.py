"""Text history -> parsed state dict for the WALL-E rule gate.

Port of WALL-E's `alfworld_runs/stateinfo_transform/state_info_transform.py`
(commit 3dad1ed, see SOURCE.md). The extractor functions (items_in_locations,
extract_item_in_hand, extract_current_position, reachable_locations,
extract_target_item) are ported as-is; they operate on a raw ReAct transcript of the
form:

    <room enumeration line: "...you see a cabinet 1, ...">
    Your task is to: ...
    > <action>
    <observation>
    > <action>
    <observation>
    ...

The only adaptation is the entry point: WALL-E parses one big transcript string sliced
at "Here is the task:". This harness instead carries `state.history` as a list of
role/content dicts (user turns prefixed "Observation: ", assistant turns emitting
"Action: <a>"). `walle_state_transform` flattens that history back into the transcript
shape above so the ported regexes run unchanged.
"""

import re
from collections import defaultdict


# --------------------------------------------------------------------- flatten
def _history_to_transcript(history):
    """Flatten this harness's state.history into WALL-E's `> action / obs` transcript.

    - The first user turn holds the instruction + ICL + the task's initial observation
      (the room enumeration + "Your task is to:"). We keep only that initial-observation
      tail (everything from the last "You are in the middle of a room" onward), so the
      ICL examples don't pollute the parse.
    - Each subsequent assistant turn contributes a `> <action>` line (the text after
      "Action:"; think/reasoning-only turns are dropped since they carry no env effect).
    - Each subsequent user turn contributes its observation text (the part after
      "Observation: ", with any appended "[World model]" push stripped).
    """
    if not history:
        return ""

    lines = []

    # Seed line(s): the initial task observation from the first user turn.
    first = str(history[0].get("content", ""))
    room_idx = first.rfind("You are in the middle of a room")
    seed = first[room_idx:] if room_idx != -1 else first
    # Cut off anything after the task line so trailing ICL/instructions don't leak in.
    task_match = re.search(r"Your task is to:.*", seed)
    if task_match:
        seed = seed[: task_match.end()]
    lines.append(seed.strip())

    # Remaining turns become the interaction transcript. Crucially, an action that the
    # WALL-E gate REJECTED in imagination never executed, so it must not enter the parse
    # (WALL-E's own loop retries and never records rejected actions). We detect a rejected
    # turn by its observation: the gate answers with "Observation: [World model] ..." and
    # does not step the env. Skip both the assistant turn and that imagination observation.
    tail = history[1:]
    for i, msg in enumerate(tail):
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "assistant":
            # Peek at the paired observation; if it's a gate rejection, this action was
            # never executed — drop it so state parsing only sees real transitions.
            nxt = tail[i + 1] if i + 1 < len(tail) else None
            if nxt is not None and nxt.get("role") == "user":
                nxt_obs = re.sub(r"^Observation:\s*", "", str(nxt.get("content", ""))).strip()
                if nxt_obs.startswith("[World model]"):
                    continue
            m = re.search(r"Action:\s?(.*)", content, re.DOTALL)
            if m:
                action = m.group(1).strip().splitlines()[0].strip()
                lines.append(f"> {action}")
        elif role == "user":
            obs = re.sub(r"^Observation:\s*", "", content).strip()
            # A gate-rejection observation is an imagination reply, not an env transition.
            if obs.startswith("[World model]"):
                continue
            # Strip any appended "[World model]" push (WM-composed env) from a real obs.
            obs = obs.split("\n[World model]")[0].strip()
            lines.append(obs)

    return "\n".join(lines)


# ----------------------------------------------------- ported extractor helpers
def extract_item_in_hand(text):
    item_in_hand = {"item_name": None, "status": None}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("> take"):
            if i + 1 < len(lines) and "Nothing happens" not in lines[i + 1]:
                match = re.search(r"take (\w+\s\d+) from", line)
                if match:
                    item_in_hand["item_name"] = match.group(1)
                    item_in_hand["status"] = "normal"
        elif line.startswith("> put"):
            if i + 1 < len(lines) and "Nothing happens" not in lines[i + 1]:
                item_in_hand["item_name"] = None
                item_in_hand["status"] = None
        elif any(line.startswith(f"> {action}") for action in ["cool", "heat", "clean"]):
            if i + 1 < len(lines) and "Nothing happens" not in lines[i + 1] and item_in_hand["item_name"] is not None:
                if line.startswith("> cool"):
                    item_in_hand["status"] = "cooled"
                elif line.startswith("> heat"):
                    item_in_hand["status"] = "heated"
                elif line.startswith("> clean"):
                    item_in_hand["status"] = "cleaned"
    return item_in_hand


def items_in_locations(text):
    target_info = defaultdict(list)
    first_line = text.split("\n")[0]
    lines = text.split("\n")
    if first_line == '':
        first_line = text.split("\n")[1]
        lines = text.split("\n")[1:]

    reachable = re.findall(r"\b(?:a\s)?(\w+\s\d+)", first_line)
    current_location = None
    for i, line in enumerate(lines[1:]):
        line = line.strip()
        if "you see" in line:
            for location in reachable:
                if location in line:
                    current_location = location
                    break
            items = re.findall(r"\b(?:a\s)?(\w+\s\d+)", line.split("you see")[1])
            if current_location:
                target_info[current_location].extend(items)
        if line.startswith("> take"):
            match = re.search(r"take (\w+\s\d+) from (\w+\s\d+)", line)
            if match:
                item = match.group(1)
                location = match.group(2)
                if item in target_info[location]:
                    if i + 2 < len(lines) and "Nothing happens" not in lines[i + 2]:
                        target_info[location].remove(item)
        if line.startswith("> put"):
            match = re.search(r"put (\w+\s\d+) in/on (\w+\s\d+)", line)
            if match:
                item = match.group(1)
                location = match.group(2)
                if i + 2 < len(lines) and "Nothing happens" not in lines[i + 2]:
                    target_info[location].append(item)
    return dict(target_info)


def extract_current_position(text):
    current_position = {"location_name": None, "status": None}
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("> go to"):
            if i + 1 < len(lines) and "Nothing happens" not in lines[i + 1]:
                match = re.search(r"go to (\w+\s\d+)", line)
                if match:
                    current_position["location_name"] = match.group(1)
                    if "closed" in lines[i + 1]:
                        current_position["status"] = "closed"
                    else:
                        current_position["status"] = "null"
        elif line.startswith("> open"):
            if i + 1 < len(lines) and "Nothing happens" not in lines[i + 1]:
                match = re.search(r"open (\w+\s\d+)", line)
                if match and match.group(1) in lines[i + 1]:
                    current_position["location_name"] = match.group(1)
                    current_position["status"] = "open"
    return current_position


def reachable_locations(text):
    first_line = text.split("\n")[0]
    if first_line == '':
        first_line = text.split("\n")[1]
    return re.findall(r"\b(?:a\s)?(\w+\s\d+)", first_line)


def extract_target_item(text):
    item_list = ["alarmclock", "butterknife", "box", "fork", "statue", "book", "dishsponge", "handtowel", "plunger", "laptop",
                 "tennisracket", "baseballbat", "remotecontrol", "bowl", "cd", "mug", "pencil", "wateringcan", "glassbottle",
                 "peppershaker", "saltshaker", "soapbar", "soapbottle", "vase", "watch", "cloth", "egg", "knife", "pot",
                 "pan", "plate", "spatula", "bread", "lettuce", "potato", "tomato", "apple", "cup", "keychain", "pillow",
                 "toiletpaper", "winebottle", "tissuebox", "spraybottle", "cellphone", "candle", "creditcard", "newspaper", "ladle"]
    lines = text.splitlines()
    target_item = None
    for line in lines:
        line = line.strip()
        if line.startswith("Your task is to: "):
            task_string = line[len("Your task is to: "):]
            words = task_string.split()
            for word in words:
                if word in item_list:
                    target_item = word
                    break
            # Adaptation: don't raise on an unmatched goal (harness must not crash).
            # WALL-E's rules never use target_item, so None is harmless.
        if target_item:
            break
    return target_item


# --------------------------------------------------------------------- entry point
def walle_state_transform(history):
    """Parse this harness's state.history into WALL-E's state dict.

    Accepts either the role/content history list (normal path) or an already-flattened
    transcript string (convenience for tests).
    """
    text = history if isinstance(history, str) else _history_to_transcript(history)
    state_info = defaultdict(list)
    state_info['target_item'] = extract_target_item(text)
    state_info['reachable_locations'] = reachable_locations(text)
    state_info['items_in_locations'] = items_in_locations(text)
    state_info['item_in_hand'] = extract_item_in_hand(text)
    state_info['current_position'] = extract_current_position(text)
    return dict(state_info)
