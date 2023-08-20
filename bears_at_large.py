
import feedparser
import random
import string
import re
import nltk
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
import requests
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd
import librosa
import soundfile as sf
import numpy as np
import torch
from functools import partial
from pathlib import Path
import argparse
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as sig
import psola
import os

#DOWNLOAD NEWS SECTION
#CHOOSE A LINK FROM THE LIST OF LISTS
def choose_random_news_link(news_links):
    # Flatten the list of lists into a single list
    flattened_links = [link for sublist in news_links for link in sublist]

    # Randomly choose a news link
    random_link = random.choice(flattened_links)

    return random_link

#MAKE A FEED FOR THE CHOSEN LINK, EXTRACT 5 HEADLINES
def extract_starter_headline(news_link):
    feed = feedparser.parse(news_link)
    entries = feed.entries

    # Check if the number of entries is less than or equal to 5
    if len(entries) <= 5:
        chosen_entries = [entry.title for entry in entries]
    else:
        random_entries = random.sample(entries, 5)
        chosen_entries = [entry.title for entry in random_entries]

    return chosen_entries

#STRING MANAGEMENT SECTION
# A MUST-CLEANUP FOR STRINGS
def clean_strings(strings):
    cleaned_strings = []

    # Define the pattern to match special symbols
    special_symbols_pattern = r"[^\w\s]"

    for s in strings:
        # Remove \xa0 tags
        cleaned_string = s.replace('\xa0', ' ')

        # Remove special symbols
        cleaned_string = re.sub(special_symbols_pattern, '', cleaned_string)

        # Remove words containing punctuation
        cleaned_string = ' '.join(word for word in cleaned_string.split() if not any(char in string.punctuation for char in word))

        # Remove punctuation from every word
        cleaned_string = cleaned_string.translate(str.maketrans('', '', string.punctuation))

        cleaned_strings.append(cleaned_string)

    return cleaned_strings

#================/\========#=========/\==================
#===============/ww\=======#========/ww\=================
#===============/Ww\=======#========/Ww\=================
#================JK========#=========JK==================

class PhraseChunker(nltk.chunk.ChunkParserI):
    def __init__(self, grammar):
        self.parser = nltk.RegexpParser(grammar)

    def parse(self, tagged_tokens):
        tree = self.parser.parse(tagged_tokens)
        return self.convert_to_phrases(tree)

    def convert_to_phrases(self, tree):
        phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() != 'S'):
            phrases.append(" ".join(word for word, tag in subtree.leaves()))
        return phrases

def extract_mashup (starter_headers_clean):
    tree_structures_list =[]
    pos_dict = {}
    for item in starter_headers_clean:
        tagged_tokens = nltk.pos_tag(nltk.word_tokenize(item))
        
        #grammar = r"""
        #    VP: {<VB.*><NP.*|PP.*|ADVP.*|ADJP.*>*}
        #    NP: {<DT|PRP\$>?<JJ>*<NN.*>+}
        #"""
        #vp_chunker = PhraseChunker(grammar)
        #vp_phrases = vp_chunker.parse(tagged_tokens)

        #np_chunker = PhraseChunker("NP: {<NN.*|PRP.*|CD>+}")
        #np_phrases = np_chunker.parse(tagged_tokens)
        #print(f"NP PRASES: {np_phrases}")

        tree = nltk.tree.Tree.fromstring("( " + " ".join([tag for word, tag in tagged_tokens]) + ")")
        tree_structures_list.append(tree.pformat())

        for word, pos in tagged_tokens:
            pos_dict[word] = pos
    return pos_dict, tree_structures_list

def compare_tree_structures_formash(pos_dict, tree_structures_list):
    mashup_list = []
    for item in tree_structures_list:
#        print(f"ITEM OF TREE STRUCTURE: {item}")
        words_nobrackets = item.translate(str.maketrans('', '', '()'))
        words = words_nobrackets.split()
#        print(f"WORDS AFTER SPLIT: {words}")
        for i in range(len(words)):
            word = words[i]
            if word in pos_dict.values():
                matching_keys = [key for key, value in pos_dict.items() if value == word]
                if matching_keys:
                    random_key = random.choice(matching_keys)
                    words[i] = random_key
                    if i == len(words) - 1:  # Fix: Change the condition to len(words) - 1
                        last_char = words[i][-1]
                        if last_char in pos_dict.values():
                            last_matching_keys = [key for key, value in pos_dict.items() if value == last_char]
                            if last_matching_keys:
                                last_random_key = random.choice(last_matching_keys)
                                words[i] = words[i][:-1] + last_random_key
                        else:
                            last_random_key = random.choice(list(pos_dict.keys()))
                            words[i] = words[i][:-1] + last_random_key
        mashup_list.append(" ".join(words))
    return mashup_list

#================/\========#=========/\==================
#===============/ww\=======#========/ww\=================
#===============/Ww\=======#========/Ww\=================
#================JK========#=========JK==================



#-EXTRACTION OF RHYMES00----------00----0-----00----
#---0-0-0-0-0--0-0------0---000------0000-----------
#----0-0-0-0-0--0-0-------0-0--00---0000------------
#-----0---0---0-----0----000-000---0000----------0--

#A request to datamuse API, rel_rly for full rhymes, rel_nry for almost rhymes
def get_rhyming_words_with_pos(word):
    parameters = {
        "rel_nry": word,
        "md": "f",
        "max": 100
    }
    response = requests.get('https://api.datamuse.com/words', params=parameters)
    rhyme = response.json()
    filtered_rhyme = [entry for entry in rhyme if float(entry['tags'][0][2:]) >= 0.9]
    filtered_words = [entry['word'] for entry in filtered_rhyme]
    tagged_words = nltk.pos_tag(filtered_words)
    return tagged_words

def generate_rhyme_list(words_list):
    rhyme_list = []
    for i in words_list:
        rhyme_to_word = get_rhyming_words_with_pos(i)
        rhyme_list.append(rhyme_to_word)
    return rhyme_list

def infuse_with_rhymes(headers_with_pos, rhyme_field_of_choice):
    rhyme_reworks = []
    used_rhymes = []

    for word, pos in headers_with_pos:
        matching_items = [(w, p) for w, p in rhyme_field_of_choice if p == pos]
        if not matching_items:
            rhyme_reworks.append(word)
        else:
            available_rhymes = [item for item in matching_items if item not in used_rhymes]
            if not available_rhymes:
                rhyme_reworks.append(word)
            else:
                random_matching_item = random.choice(available_rhymes)
                rhyme_reworks.append(random_matching_item[0])
                used_rhymes.append(random_matching_item)

    return rhyme_reworks

#-00000000------------00----------00----0-----00----
#---0-0-0-0-0--0-0------0---000------0000-----------
#----0-0-0-0-0--0-0-------0-0--00---0000------------
#-----0---0---0-----0----000-000---0000----------0--

#mannipulation to put together headers and their phonetic representation
def combine_lists(list1, list2):
    combined_list = []
    for item1, item2 in zip(list1, list2):
        split_item1 = item1.split()
        split_item2 = item2.split()
        sublist = [(split_item1[i], split_item2[i]) for i in range(len(split_item1))]
        combined_list.extend(sublist)
    return combined_list

#split the eventual poem into 6-word lines
def join_with_newline(strings):
    joined_string = ''
    for i, string in enumerate(strings, 1):
        joined_string += string
        if i % 6 == 0:
            joined_string += '\n'
        else:
            joined_string += ' '
    return joined_string.strip()


#LET'S GET THE INITIAL HEADLINES FOR REWORKING

def get_starter_headers ():
    lst_of_news = [
        ['https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://www.theguardian.com/world/rss',
        'https://www.aljazeera.com/xml/rss/all.xml',
        'https://www.bbc.com/news',
        'https://www.nytimes.com/',
        'https://www.theguardian.com/international',
        'https://edition.cnn.com/'],
        ['https://www.rt.com/',
        'https://sputniknews.com/',
        'https://tass.com/',
        'https://www.themoscowtimes.com/'
        'https://www.rbth.com/',
        'https://russia-insider.com/',
        'https://meduza.io/en'
        'https://sptimes.ru/'],
        ['https://timesofindia.indiatimes.com/',
        'https://indianexpress.com/',
        'https://www.hindustantimes.com/',
        'https://www.ndtv.com/',
        'https://www.thehindu.com/',
        'https://www.indiatoday.in/', 
        'https://www.firstpost.com/', 
        'https://economictimes.indiatimes.com/', 
        'https://scroll.in/', 
        'https://www.news18.com/']
    ]

    #CHOOSE A LINK FROM A LIST OF LINKS TO EXTRACT NEWS
    random_news_link = choose_random_news_link(lst_of_news)

    #EXTRACT 5 HEADLINES INTO A LIST
    raw_starter_headers = extract_starter_headline(random_news_link)
        #clean the starter headers
    while not raw_starter_headers:
        random_news_link = choose_random_news_link(lst_of_news)
        raw_starter_headers = extract_starter_headline(random_news_link)

    #CLEAN THE LIST
    #cleaning up speciial symbols, punctuation etc
    starter_headers_clean = clean_strings(raw_starter_headers)
    return starter_headers_clean

#========================================================
#============MAKE SOME POETIC MAGIC======================
def make_a_verse():

    starter_headers_clean = get_starter_headers()

    pos_dict, tree_structures_list = extract_mashup(starter_headers_clean)

    mashup_list = compare_tree_structures_formash(pos_dict, tree_structures_list)

    #-----====Here we have named entities and the mashup_list. Time to find rhymes for it all

    total_rhyme_dict = {}

    for phrase in mashup_list:
        words_list = phrase.split()
        rhyme_list = generate_rhyme_list(words_list)
        total_rhyme_dict[phrase] = words_list

    while not rhyme_list:
        rhyme_list = generate_rhyme_list(words_list)

    #==========================
    #+++++++++Generate rhymes++
    #==========================

    #it will be a combined list of rhyming words extracted with Datamuse API
    rhyme_field_of_choice = []
    for i in rhyme_list:
        rhyme_field_of_choice.extend(i)
    #make a list of headlines parallel to their pos tags for every word
    headers_with_pos = combine_lists(starter_headers_clean, tree_structures_list)
    #list for output of rhyming words
    rhyme_reworks = []
    #replace words that do have relevant rhymes with rhymes
    rhyme_reworks = infuse_with_rhymes(headers_with_pos, rhyme_field_of_choice)
    # Put lines together with new line after every 6 words
    composed_poem = join_with_newline(rhyme_reworks)
    return composed_poem


#PRONOUNCING THE LINES====================&&&
#========================================777=
#============================START======rrrr=


def voice_it_over(text):
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/fastspeech2-en-ljspeech",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)

    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)

    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, models[0], generator, sample)

    return wav, rate


#PRONOUNCING THE LINES====================&&&
#========================================777=
#==============================END======rrrr=

SEMITONES_IN_OCTAVE = 12


def degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    # To properly perform pitch rounding to the nearest degree from the scale, we need to repeat
    # the first degree raised by an octave. Otherwise, pitches slightly lower than the base degree
    # would be incorrectly assigned.
    degrees = np.concatenate((degrees, [degrees[0] + SEMITONES_IN_OCTAVE]))
    return degrees


def closest_pitch(f0):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(f0))
    # To preserve the nan values.
    nan_indices = np.isnan(f0)
    midi_note[nan_indices] = np.nan
    # Convert back to Hz.
    return librosa.midi_to_hz(midi_note)


def closest_pitch_from_scale(f0, scale):
    """Return the pitch closest to f0 that belongs to the given scale"""
    # Preserve nan.
    if np.isnan(f0):
        return np.nan
    degrees = degrees_from(scale)
    midi_note = librosa.hz_to_midi(f0)
    # Subtract the multiplicities of 12 so that we have the real-valued pitch class of the
    # input pitch.
    degree = midi_note % SEMITONES_IN_OCTAVE
    # Find the closest pitch class from the scale.
    degree_id = np.argmin(np.abs(degrees - degree))
    # Calculate the difference between the input pitch class and the desired pitch class.
    degree_difference = degree - degrees[degree_id]
    # Shift the input MIDI note number by the calculated difference.
    midi_note -= degree_difference
    # Convert to Hz.
    return librosa.midi_to_hz(midi_note)


def aclosest_pitch_from_scale(f0, scale):
    """Map each pitch in the f0 array to the closest pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(f0)
    for i in np.arange(f0.shape[0]):
        sanitized_pitch[i] = closest_pitch_from_scale(f0[i], scale)
    # Perform median filtering to additionally smooth the corrected pitch.
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    # Remove the additional NaN values after median filtering.
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = \
        sanitized_pitch[np.isnan(smoothed_sanitized_pitch)]
    return smoothed_sanitized_pitch


def autotune(audio, sr, correction_function, target_f0, plot=False):
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    # Apply the chosen adjustment strategy to the pitch.
    corrected_f0 = correction_function(target_f0)

    if plot:
        # Plot the spectrogram, overlaid with the original pitch trajectory and the adjusted
        # pitch trajectory.
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        time_points = librosa.times_like(stft, sr=sr, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(log_stft, x_axis='time', y_axis='log', ax=ax, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.plot(time_points, target_f0, label='original pitch', color='cyan', linewidth=2)
        ax.plot(time_points, corrected_f0, label='corrected pitch', color='orange', linewidth=1)
        ax.legend(loc='upper right')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.savefig('pitch_correction.png', dpi=300, bbox_inches='tight')

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)



def main():
    autotune_sources_dir = "C:/Users/User/Documents/Python Scripts/hda/autotune_sources"
    autotune_sources = os.listdir(autotune_sources_dir)

    for i in range(5):
        verse = make_a_verse()
        print(verse)
        voiced_text, sample_rate = voice_it_over(verse)

        # Select a random target audio file from the autotune_sources directory.
        target_file = random.choice(autotune_sources)
        target_track = os.path.join(autotune_sources_dir, target_file)

        # Load the target audio file.
        target_audio, target_sr = librosa.load(target_track, sr=None, mono=True)
        target_f0, _, _ = librosa.pyin(target_audio, sr=target_sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Convert the tensor to a numpy array
        sample_audio = voiced_text.numpy()

        # Only mono-files are handled. If stereo files are supplied, only the first channel is used.
        if sample_audio.ndim > 1:
            sample_audio = sample_audio[0, :]

        # Pick the pitch adjustment strategy according to the arguments.
        correction_function = partial(aclosest_pitch_from_scale, scale="D#:min")

        # Perform the auto-tuning.
        pitch_corrected_audio = autotune(sample_audio, sample_rate, correction_function, target_f0)

        # Write the corrected audio to an output file.
        output_file = f"C:/Users/User/Documents/Python Scripts/hda/audio_outputs/py1_voice_output{i+1}.wav"
        sf.write(output_file, pitch_corrected_audio, sample_rate)


if __name__ == '__main__':
    main()
