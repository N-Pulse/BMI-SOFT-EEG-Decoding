#%%

from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet
import random

def main():
    info = StreamInfo(name='stimulus_stream', type='Markers', channel_count=1,
                      channel_format='int32', source_id='stimulus_stream_001')
    outlet = StreamOutlet(info)  # Broadcast the stream.
    
    markers = {
        'test': [99], 
        'start': [88],
        'baseline' : [77], 
        # elbow flexion, elbow extension, forearm supination, forearm pronation, hand close, and hand open
        'elbow_flexion' : [1],
        'elbow_extension' : [2],
        'forearm_supination' : [3],
        'forearm_pronation' : [4],
        'hand_close' : [5],
        'hand_open' : [6],
    }

    win = visual.Window(fullscr=True, allowGUI=False, monitor='testMonitor', # [1000, 800]
                         units='deg', color="white")

    # baseline
    base = visual.TextStim(win, text=" ", color="white")

    # define movements
    # mov1 = visual.TextStim(win, text="**open hand**")
    # mov2 = visual.TextStim(win, text="**close hand**")
    # mov3 = visual.TextStim(win, text="**movement 3**")

    cross = visual.ImageStim(win, image="Experiment/fixation_cross.png", size=(15,15))  # Adjust size as needed
    ef = visual.ImageStim(win, image="Experiment/elbow_flexion.png")  # Adjust size as needed  
    ee = visual.ImageStim(win, image="Experiment/elbow_extension.png")  # Adjust size as needed
    fs = visual.ImageStim(win, image="Experiment/forearm_supination.png")  # Adjust size as needed
    fp = visual.ImageStim(win, image="Experiment/forearm_pronation.png")  # Adjust size as needed
    hc = visual.ImageStim(win, image="Experiment/hand_close.png")  # Adjust size as needed
    ho = visual.ImageStim(win, image="Experiment/hand_open.png")  # Adjust size as needed

    mov_done = visual.TextStim(win, color="black", text="Relax. \n\n If ready for the next movement, please press <SPACE>")

    countdown = visual.TextStim(win, text="3", height=3.0)

    # def cd_stim():
    #     for t in range(3, 0, -1):  # Countdown from 3 to 1
    #         countdown.text = str(f'{t}')
    #         countdown.draw()
    #         core.wait(0.9)
    #         win.flip() # clears the screen
    #         core.wait(.1)
    #     core.wait(1)
        
    def mov_stim(movi, mov_n):
        # show which movement is coming up
        # hint = visual.TextStim(win, color="white")
        # hint.text = f'Next movement is: \n\n     {movi.text} \n\n Are you ready? Press <SPACE>'
        # hint.draw()

        core.wait(1.5)

        # show the fixation cross
        cross.draw()
        win.flip()

        core.wait(0.5)
        win.flip()

        # wait for 2 seconds
        core.wait(2)
        # key_resp()

        # start countdown
        # cd_stim()

        # wait for 2 seconds
        

        # show movement & send marker for performed movement
        movi.draw()
        win.flip()
        outlet.push_sample(mov_n)

        core.wait(3)
        win.flip()
        core.wait(1)

        # movement done
        mov_done.draw()
        win.flip()
        key_resp()
        win.flip()

    def key_resp():
        keys = event.waitKeys(keyList=['space', 'escape'])
        win.flip()
        if 'escape' in keys:
            win.close()
            core.quit()

    # Send triggers to test communication
    for _ in range(5):
        outlet.push_sample(markers['test'])
        core.wait(0.5)

    # Start the recording
    start = visual.TextStim(win, color="black", text="To start the recording, hit the record button on labrecorder and press <SPACE>")
    start.draw()
    outlet.push_sample(markers['start'])
    win.flip()
    key_resp()
    
    # Show stimuli -- random order!!
    # Define the stimuli and markers
    stimuli = [
        (base, markers['baseline']),
        (ef, markers['elbow_flexion']),
        (ee, markers['elbow_extension']),
        (fs, markers['forearm_supination']),
        (fp, markers['forearm_pronation']),
        (ho, markers['hand_open']),
        (hc, markers['hand_close'])
    ]  

    # Repeat each stimulus 6 times
    repeated_stimuli = stimuli * 7  # Creates 42 total (7 * 6)

    # Shuffle the list randomly
    random.shuffle(repeated_stimuli)

    for stimulus, marker in repeated_stimuli:
        mov_stim(stimulus, marker)

    # for _ in range(4):
    #     # stimulus, marker = random.choice(stimuli)  # Randomly select one stimulus and its corresponding marker
    #     mov_stim(stimulus, marker)

    # mov_stim(mov1, markers['movement1'])

    # mov_stim(mov3, markers['movement3'])

    # mov_stim(mov2, markers['movement2'])
    
    win.flip()
    end = visual.TextStim(win, text="Finished! Thank you")
    end.draw()
    win.flip()
    key_resp()

    win.close()
    core.quit()

if __name__ == "__main__":
    main()

#%%