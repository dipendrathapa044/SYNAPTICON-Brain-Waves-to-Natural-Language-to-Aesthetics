import time
import csv
import numpy as np
import pygame
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Configuration 
use_synthetic = False  # Set True for testing without real hardware
serial_port = "/dev/cu.usbserial-DP05I4NP"
session_index = 30      # Session number (for file naming, if running multiple sessions)

# BrainFlow: Initialize board connection
BoardShim.enable_board_logger()  
params = BrainFlowInputParams()
if use_synthetic:
    board_id = BoardIds.SYNTHETIC_BOARD.value
    params.serial_port = ""  
else:
    board_id = BoardIds.CYTON_DAISY_BOARD.value  # Cyton+Daisy 16-ch board ID
    params.serial_port = serial_port             # set the port for OpenBCI dongle

board = BoardShim(board_id, params)
board.prepare_session()
print("Connecting to the OpenBCI board...")
board.start_stream()  
print("EEG stream started.")

# Pygame: Initialize full-screen window for text display
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Sentence Presentation")
bg_color = (0, 0, 0)             
text_color = (255, 255, 255)     
font = pygame.font.Font(None, 100)  
screen_rect = screen.get_rect()

# Define helper functions for display
def draw_countdown(seconds):
    """Display a countdown timer (in seconds) on the screen."""
    for remaining in range(seconds, 0, -1):
        screen.fill(bg_color)
        text_surf = font.render(f"Starting in: {remaining}", True, text_color)
        text_rect = text_surf.get_rect(center=screen_rect.center)
        screen.blit(text_surf, text_rect)
        pygame.display.flip()
        time.sleep(1)
        for event in pygame.event.get():  
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                pygame.quit()
                board.stop_stream()
                board.release_session()
                raise SystemExit("Experiment aborted by user.")

def show_fixation(duration=2.0):
    """Show a fixation cross for the given duration (in seconds)."""
    screen.fill(bg_color)
    cross = font.render("+", True, text_color)
    cross_rect = cross.get_rect(center=screen_rect.center)
    screen.blit(cross, cross_rect)
    pygame.display.flip()
    start_time = time.time()
    while time.time() - start_time < duration:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                pygame.quit()
                board.stop_stream()
                board.release_session()
                raise SystemExit("Experiment aborted by user.")
        time.sleep(0.01)

# Load sentences from CSV file
sentences = []
with open("data/synapticon_sentences.csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  
    for row in reader:
        if row:  
            sentences.append(row[0])
print(f"{len(sentences)} sentences loaded for the task.")

# Pre-experiment countdown (10 seconds)
draw_countdown(10)

# Main Recording Loop (iterate through each sentence)
sample_rate = BoardShim.get_sampling_rate(board_id)  # sampling rate for the board (e.g., 125 Hz for Cyton+Daisy)
eeg_channel_indices = BoardShim.get_eeg_channels(board_id)  # indices of EEG channels in BrainFlow data array
session_data = []  # will collect [time, ch1, ch2, ..., ch16, label] for each sample

for sentence_index, sentence in enumerate(sentences):
    # Display the sentence word-by-word for 10 seconds
    words = sentence.split()
    if len(words) == 0:
        continue
    # Pre-render word surfaces for smooth display
    word_surfs = [font.render(word, True, text_color) for word in words]
    # Compute positions to center the entire sentence on screen
    total_width = sum(surf.get_width() for surf in word_surfs) + (len(words) - 1) * font.size(" ")[0]
    start_x = (screen_rect.width - total_width) // 2
    y = screen_rect.centery
    positions = []
    x = start_x
    for surf in word_surfs:
        positions.append((x, y - surf.get_height() // 2))
        x += surf.get_width() + font.size(" ")[0]

    sentence_start_time = time.time()
    # Calculate time per word to fill the 10-second sentence duration
    sentence_duration = 10.0
    time_per_word = sentence_duration / len(words)

    # Flush any old data from the buffer at the start of the sentence segment
    board.get_board_data()  

    # Sequentially reveal words one by one
    for i in range(len(words)):
        word_start = time.time()
        while time.time() - word_start < time_per_word:
            # Display all words up to index i (so words accumulate on screen)
            screen.fill(bg_color)
            for j in range(i + 1):
                screen.blit(word_surfs[j], positions[j])
            pygame.display.flip()
            # Check for exit key
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                    pygame.quit()
                    board.stop_stream()
                    board.release_session()
                    raise SystemExit("Experiment aborted by user.")
            time.sleep(0.01)
    # Ensure the full sentence remains on screen for the remainder of the 10s
    while time.time() - sentence_start_time < sentence_duration:
        for event in pygame.event.get():  
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                pygame.quit()
                board.stop_stream()
                board.release_session()
                raise SystemExit("Experiment aborted by user.")
        time.sleep(0.01)

    # At this point, 10 seconds have passed for this sentence.
    # Retrieve the EEG data for this 10-second interval
    expected_samples = int(sentence_duration * sample_rate)
    data = board.get_board_data()  # get all data in buffer (which should correspond to the last ~12s: 10s sentence + ~2s fixation)
    # Note: get_board_data() clears the buffer after reading.
    # We will separate sentence vs fixation data below.
    if data.shape[1] == 0:
        print("No EEG data captured for sentence index", sentence_index)
    else:
        # BrainFlow returns data as a 2D array [channels x samples].
        # Extract EEG channels (e.g., 16 channels for Cyton+Daisy)
        eeg_data = data[eeg_channel_indices, :].T  # shape: (samples, 16)
        # If we captured more than the sentence period (some fixation data), trim it
        if eeg_data.shape[0] > expected_samples:
            eeg_data = eeg_data[:expected_samples, :]
        # If less (in case of slight timing differences), you can pad or handle accordingly.
        num_samples = eeg_data.shape[0]
        # Generate time stamps for these samples relative to sentence start (0 to ~10s)
        times = np.arange(0, num_samples) / float(sample_rate)
        # Create label array for this segment
        labels = np.full((num_samples, 1), sentence_index)
        # Combine time, EEG channels, and label into one array
        segment_array = np.hstack((times.reshape(-1, 1), eeg_data, labels))
        session_data.append(segment_array)
    # Show the fixation cross (2 seconds), during which we are not labeling EEG
    show_fixation(duration=2.0)
    # After fixation, loop continues to next sentence (the board continues streaming through fixation, 
    # but we've cleared the buffer at sentence start and after get_board_data above, 
    # so fixation data is effectively not included in session_data).

# End of all sentences
pygame.quit()
# Stop and release the EEG board
board.stop_stream()
board.release_session()
print("Session recording complete. Saving data...")

# Save the collected session data to CSV
import numpy as np
import pandas as pd
if session_data:
    session_data = np.vstack(session_data)  # combine all segments vertically
    # Prepare column names: time, ch1...ch16, label
    num_channels = len(eeg_channel_indices)
    cols = ["time"] + [f"ch{i+1}" for i in range(num_channels)] + ["label"]
    df = pd.DataFrame(session_data, columns=cols)
    out_path = f"data/raw/session_{session_index:02d}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved session data to session_{session_index}.csv (shape: {session_data.shape})")
else:
    print("No data collected to save.")
