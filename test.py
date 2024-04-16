from music21 import *

file_path = "example/score_inputs/test.musicxml"

# MusicXMLファイルを読み込む
score = converter.parse(file_path)

for element in score.recurse():
    if isinstance(element, note.Note):
        print(element.lyrics[0].text)
        print("Note:", element.nameWithOctave)
    elif isinstance(element, chord.Chord):
        print("Chord:", ' '.join(n.nameWithOctave for n in element.pitches))
    elif isinstance(element, stream.Measure):
        print("Measure:", element.number)