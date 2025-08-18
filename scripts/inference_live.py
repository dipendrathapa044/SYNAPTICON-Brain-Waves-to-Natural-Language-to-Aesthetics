import argparse
import time
import csv
import numpy as np
import pandas as pd
import pygame
import torch
from pathlib import Path
from transformers import GPT2Tokenizer
from eeg_cnn_decoder import EEG2Text
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy import signal

# EEG Preprocessing
FS = 125          # sampling rate (Hz)
NOTCH_FREQ = 50   # line noise freq
Q = 30            # notch quality
BP = (1, 40)      # band-pass limits

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype="band")

def preprocess(eeg_window: np.ndarray):
    #Apply notch and band-pass to raw window (time, channels).
    x = eeg_window.T.astype(np.float32)  # (ch, time)
    b_n, a_n = signal.iirnotch(NOTCH_FREQ, Q, FS)
    x = signal.filtfilt(b_n, a_n, x, axis=1)
    b_bp, a_bp = butter_bandpass(BP[0], BP[1], FS)
    x = signal.filtfilt(b_bp, a_bp, x, axis=1)
    return x.T  # back to (time, ch)

# Pygame Presentation

def draw_countdown(screen, font, screen_rect, seconds, bg_color, text_color, board):
    for remaining in range(seconds, 0, -1):
        screen.fill(bg_color)
        surf = font.render(f"Starting in: {remaining}", True, text_color)
        rect = surf.get_rect(center=screen_rect.center)
        screen.blit(surf, rect)
        pygame.display.flip()
        time.sleep(1)
        for evt in pygame.event.get():
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_e:
                pygame.quit(); board.stop_stream(); board.release_session()
                raise SystemExit("Experiment aborted by user.")

def show_fixation(screen, font, screen_rect, duration, bg_color, text_color, board):
    screen.fill(bg_color)
    cross = font.render("+", True, text_color)
    cross_rect = cross.get_rect(center=screen_rect.center)
    screen.blit(cross, cross_rect)
    pygame.display.flip()
    start = time.time()
    while time.time() - start < duration:
        for evt in pygame.event.get():
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_e:
                pygame.quit(); board.stop_stream(); board.release_session()
                raise SystemExit("Experiment aborted by user.")
        time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sentences_csv', default='data/synapticon_sentences.csv')
    parser.add_argument('--output', default='outputs/live_inference.txt')
    parser.add_argument('--serial_port', required=True)
    parser.add_argument('--board_id', default='CYTON_DAISY_BOARD')
    parser.add_argument('--window_sec', type=float, default=10.0)
    parser.add_argument('--fix_sec', type=float, default=2.0)
    args = parser.parse_args()

    # Prepare BrainFlow board
    params = BrainFlowInputParams(); params.serial_port = args.serial_port
    board_id = BoardIds[args.board_id].value
    board = BoardShim(board_id, params)
    BoardShim.enable_dev_board_logger()
    board.prepare_session(); board.start_stream(450000)
    sample_rate = BoardShim.get_sampling_rate(board_id)
    eeg_idx = BoardShim.get_eeg_channels(board_id)

    # Load model + tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = EEG2Text(
        vocab_size=tokenizer.vocab_size, max_len=128,
        d_model=512, nhead=8, num_layers=6, dim_feedforward=2048,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    font = pygame.font.Font(None, 100)
    screen_rect = screen.get_rect()
    bg_color, text_color = (0,0,0), (255,255,255)

    # Load sentences
    with open(args.sentences_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f); header = next(reader, None)
        sentences = [row[0] for row in reader if row]

    # Pre-experiment countdown
    draw_countdown(screen, font, screen_rect, 10, bg_color, text_color, board)

    # Open output log
    fout = open(args.output, 'a')

    # Main loop: for each sentence
    for idx, sentence in enumerate(sentences):
        # Prepare word surfaces + positions
        words = sentence.split()
        word_surfs = [font.render(w,True,text_color) for w in words]
        space = font.size(' ')[0]
        total_w = sum(s.get_width() for s in word_surfs) + (len(words)-1)*space
        start_x = (screen_rect.width-total_w)//2
        y = screen_rect.centery
        positions = []
        x = start_x
        for surf in word_surfs:
            positions.append((x, y - surf.get_height()//2))
            x += surf.get_width() + space

        # Clear buffer, then display sentence word-by-word over window_sec
        board.get_board_data()
        start_time = time.time()
        for i in range(len(words)):
            t0 = time.time()
            while time.time() - t0 < args.window_sec/len(words):
                screen.fill(bg_color)
                for j in range(i+1):
                    screen.blit(word_surfs[j], positions[j])
                pygame.display.flip()
                for evt in pygame.event.get():
                    if evt.type==pygame.KEYDOWN and evt.key==pygame.K_e:
                        pygame.quit(); board.stop_stream(); board.release_session()
                        raise SystemExit
                time.sleep(0.01)
        # Ensure full sentence stays on screen until 10s
        while time.time() - start_time < args.window_sec:
            pygame.display.flip()
            time.sleep(0.01)

        # Grab EEG data for this window
        data = board.get_board_data()  # all buffered
        eeg_data = data[eeg_idx, :].T
        # Trim or pad to exact samples
        exp_samps = int(args.window_sec*sample_rate)
        if eeg_data.shape[0] > exp_samps:
            eeg_data = eeg_data[:exp_samps]
        elif eeg_data.shape[0] < exp_samps:
            pad = np.zeros((exp_samps - eeg_data.shape[0], len(eeg_idx)), np.float32)
            eeg_data = np.vstack([eeg_data, pad])

        # Preprocess + inference
        proc = preprocess(eeg_data)  # (samples, ch)
        window = proc.T.copy().astype(np.float32)
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)

        # Constrain generation length to original sentence length
        tgt_tokens = tokenizer.tokenize(sentence)
        max_gen = len(tgt_tokens) + 2  # allow room for EOS
        tok_ids = model.generate(
            tensor,
            tokenizer,
            max_new_tokens=max_gen,
        )
        text = tokenizer.decode(tok_ids, skip_special_tokens=True)

        # Log result
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"{ts}: [{idx}] {text}"
        print(line, end=''); fout.write(line); fout.flush()

        # Show fixation
        show_fixation(screen, font, screen_rect, args.fix_sec, bg_color, text_color, board)

    # Cleanup
    fout.close()
    pygame.quit()
    board.stop_stream()
    board.release_session()
    print("Session complete.")
