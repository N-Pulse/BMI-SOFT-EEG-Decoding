# -*- coding: utf-8 -*-
from psychopy import core, visual, event, gui
from pylsl import StreamInfo, StreamOutlet
import os
import random

# ----------------------------- Config -----------------------------

IMG_DIR = "Experiment"  # folder containing your PNGs

MOVEMENTS = {
    "elbow_flexion": {
        "marker": 1,
        "image": os.path.join(IMG_DIR, "elbow_flexion.png"),
        "label": "Elbow Flexion",
    },
    "elbow_extension": {
        "marker": 2,
        "image": os.path.join(IMG_DIR, "elbow_extension.png"),
        "label": "Elbow Extension",
    },
    "forearm_supination": {
        "marker": 3,
        "image": os.path.join(IMG_DIR, "forearm_supination.png"),
        "label": "Forearm Supination",
    },
    "forearm_pronation": {
        "marker": 4,
        "image": os.path.join(IMG_DIR, "forearm_pronation.png"),
        "label": "Forearm Pronation",
    },
    "hand_close": {
        "marker": 5,
        "image": os.path.join(IMG_DIR, "hand_close.png"),
        "label": "Hand Close",
    },
    "hand_open": {
        "marker": 6,
        "image": os.path.join(IMG_DIR, "hand_open.png"),
        "label": "Hand Open",
    },
    # Optional baseline (no image)
    "baseline": {
        "marker": 77,
        "image": None,
        "label": "Baseline",
    },
}

def safe_wait(seconds: float):
    """Wait while polling for Esc to quit early."""
    t = core.Clock()
    while t.getTime() < seconds:
        if 'escape' in event.getKeys(keyList=['escape']):
            core.quit()
        core.wait(0.01)

def load_or_text_stim(win, image_path, fallback_text):
    """Try to load an ImageStim; if missing, return a TextStim instead."""
    if image_path and os.path.exists(image_path):
        return visual.ImageStim(win, image=image_path)
    else:
        return visual.TextStim(win, text=fallback_text, color="black", height=2.0)

# ----------------------------- GUI -----------------------------------------

def setup_via_gui():
    # We’ll offer checkboxes for each movement + baseline.
    d = gui.Dlg(title="Experiment Setup")
    d.addText("Select movements (Yes = include)")
    include_defaults = {
        "elbow_flexion": "Yes",
        "elbow_extension": "Yes",
        "forearm_supination": "Yes",
        "forearm_pronation": "Yes",
        "hand_close": "Yes",
        "hand_open": "Yes",
        "baseline": "No",  # off by default
    }
    for key in ["elbow_flexion","elbow_extension","forearm_supination",
                "forearm_pronation","hand_close","hand_open","baseline"]:
        d.addField(MOVEMENTS[key]["label"], choices=["Yes","No"], tip=f"Include {MOVEMENTS[key]['label']}?", initial=include_defaults[key])

    d.addText("Timing")
    d.addField("Cue duration (s)", 3.0)
    d.addField("Inter-cue interval (s)", 2.0)
    d.addField("Fixation before cue (s)", 1.0)
    d.addField("Countdown before first cue (s)", 0)  # 0 = off

    d.addText("Design")
    d.addField("# cues per movement", 6)
    d.addField("Order", choices=["Random", "Ordered"], tip="Random shuffles all cues")
    d.addField("Test LSL pings first?", choices=["Yes","No"], initial="Yes")
    d.addField("Full screen?", choices=["Yes","No"], initial="Yes")

    ok = d.show()
    if d.OK is False:
        core.quit()

    # Parse GUI output
    idx = 0
    included = {}
    for key in ["elbow_flexion","elbow_extension","forearm_supination",
                "forearm_pronation","hand_close","hand_open","baseline"]:
        included[key] = (ok[idx] == "Yes")
        idx += 1

    cue_dur = float(ok[idx]); idx += 1
    iti = float(ok[idx]); idx += 1
    fix_dur = float(ok[idx]); idx += 1
    countdown_s = float(ok[idx]); idx += 1

    n_per = int(ok[idx]); idx += 1
    order_choice = ok[idx]; idx += 1
    do_test = (ok[idx] == "Yes"); idx += 1
    fullscreen = (ok[idx] == "Yes"); idx += 1

    # Basic sanity
    if cue_dur <= 0: cue_dur = 3.0
    if iti < 0: iti = 0
    if fix_dur < 0: fix_dur = 0
    if n_per < 1: n_per = 1

    return {
        "included": included,
        "cue_dur": cue_dur,
        "iti": iti,
        "fix_dur": fix_dur,
        "countdown_s": countdown_s,
        "n_per": n_per,
        "order_choice": order_choice,
        "do_test": do_test,
        "fullscreen": fullscreen
    }

# ----------------------------- Main ----------------------------------------

def main():
    # LSL outlet
    info = StreamInfo(name='stimulus_stream', type='Markers', channel_count=1,
                      channel_format='int32', source_id='stimulus_stream_001')
    outlet = StreamOutlet(info)

    # GUI selections
    cfg = setup_via_gui()

    # Window
    win = visual.Window(fullscr=cfg["fullscreen"], allowGUI=False, monitor='testMonitor',
                        units='deg', color="white")
    cross = visual.ImageStim(win, image=os.path.join(IMG_DIR, "fixation_cross.png"), size=(15,15)) \
            if os.path.exists(os.path.join(IMG_DIR, "fixation_cross.png")) else \
            visual.TextStim(win, text="+", color="black", height=4.0)
    ready_txt = visual.TextStim(win, color="black", text="Start your LabRecorder now.\n\nRecording will begin shortly...")
    end_txt = visual.TextStim(win, color="black", text="Finished! Thank you.")
    countdown = visual.TextStim(win, text="", height=3.0, color="black")

    # Build the stimulus objects (image or text fallback)
    stim_objects = {}
    for key, spec in MOVEMENTS.items():
        stim_objects[key] = load_or_text_stim(win, spec["image"], spec["label"])

    # Optional test pings before starting
    if cfg["do_test"]:
        test_marker = 99
        for _ in range(5):
            outlet.push_sample([test_marker])
            safe_wait(0.5)

    # Start prompt (auto-continue)
    ready_txt.draw()
    win.flip()
    # Send 'start' marker
    outlet.push_sample([88])
    safe_wait(1.5)

    # Optional countdown
    if cfg["countdown_s"] > 0:
        for t in range(int(cfg["countdown_s"]), 0, -1):
            countdown.text = str(t)
            countdown.draw()
            win.flip()
            safe_wait(1.0)

    # Build trial list
    trials = []
    for key, include in cfg["included"].items():
        if include:
            for _ in range(cfg["n_per"]):
                trials.append(key)

    if len(trials) == 0:
        # Nothing selected; bail politely
        msg = visual.TextStim(win, text="No movements selected.\n\nExiting.", color="black")
        msg.draw(); win.flip(); safe_wait(2.0)
        win.close(); core.quit()

    if cfg["order_choice"].lower().startswith("random"):
        random.shuffle(trials)

    # Trial loop
    for idx, key in enumerate(trials, start=1):
        # Pre-cue fixation
        if cfg["fix_dur"] > 0:
            cross.draw()
            win.flip()
            safe_wait(cfg["fix_dur"])

        # Cue on
        stim_objects[key].draw()
        win.flip()

        # Send marker at cue onset
        outlet.push_sample([MOVEMENTS[key]["marker"]])

        # Hold cue
        safe_wait(cfg["cue_dur"])

        # Inter-cue interval (blank screen)
        if cfg["iti"] > 0:
            win.flip(clearBuffer=True)
            safe_wait(cfg["iti"])

    # End screen
    win.flip(clearBuffer=True)
    end_txt.draw()
    win.flip()
    safe_wait(2.0)

    win.close()
    core.quit()

# ----------------------------- Run -----------------------------------------
if __name__ == "__main__":
    main()
