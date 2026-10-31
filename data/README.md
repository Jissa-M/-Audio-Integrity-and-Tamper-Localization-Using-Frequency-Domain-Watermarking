# Dataset

All clips are 20 s, mono, 16 kHz WAV, cut from longer recordings.
Raw downloads (MP3/OGG/FLAC) are **not** committed. Only the cut clips are
(about 640 KB each, 12 MB total).

## Rebuild the test set

```bash
pip install numpy scipy soundfile librosa matplotlib pandas pytest
ffmpeg -version          # ffmpeg with libmp3lame is required
python tamper_simulation.py
```

This regenerates `data/generated/` (about 192 files, not committed) and
`results/ground_truth.csv` (204 rows) from the clips, the key and the seed
(2026). Files with two splices have two rows with the same `file_name`.

## Split (by source recording; no recording appears in two groups)

| Split | Sources |
|---|---|
| training | src01, src05, src09, src10 |
| validation | src02, src06, src03, src11 |
| test | src04, src07, src08, src12 |

## Speech (LibriVox, public domain)

LibriVox recordings are dedicated to the public domain. All authors died
before 1966. "Section" is the LibriVox section number, which may differ
from the printed chapter number. Clips were cut from 60 s to 80 s of each section.

| Clip | Role | Book (author) | Section | Reader | Gender | URL |
|---|---|---|---|---|---|---|
| src01 | source | Pride and Prejudice (Austen) | 6 | Kristen McQuillin | F | https://librivox.org/pride-and-prejudice-by-jane-austen/ |
| src02 | source | Three Men in a Boat (Jerome K. Jerome) | 3 | Betsie Bush | F | https://librivox.org/three-men-in-a-boat-by-jerome-k-jerome/ |
| src03 | source | Alice's Adventures in Wonderland (Carroll) | 7 | Kara Shallenberg | F | https://librivox.org/alices-adventures-in-wonderland-by-lewis-carroll-4/ |
| src04 | source | Wuthering Heights (E. Brontë) | 1 | Lady Maria | F | https://librivox.org/wuthering-heights-by-emily-bronte/ |
| src05 | source | Great Expectations (Dickens) | 3 | Mark F. Smith | M | https://librivox.org/great-expectations-by-charles-dickens/ |
| src06 | source | Memoirs of Sherlock Holmes (Conan Doyle) | 9 | Martin Clifton | M | https://librivox.org/the-memoirs-of-sherlock-holmes-by-sir-arthur-conan-doyle/ |
| src07 | source | The Invisible Man (H. G. Wells) | 13 | Alex Foster | M | https://librivox.org/invisible-man-by-h-g-wells/ |
| src08 | source | Robinson Crusoe (Defoe) | 7 | Denny Sayers | M | https://librivox.org/robinson-crusoe-by-daniel-defoe/ |
| src09 | source | Treasure Island (Stevenson) | 14 | Mark Bradford | M | https://librivox.org/treasure-island-by-robert-louis-stevenson/ |
| donor01 | donor | Dorothy and the Wizard in Oz (Baum) | 5 | Phil Chenevert | M | https://librivox.org/dorothy-and-the-wizard-in-oz-by-l-frank-baum-2/ |
| donor02 | donor | The Time Machine (H. G. Wells) | 1 | Aaron Elliott | M | https://librivox.org/the-time-machine-by-hg-wells/ |
| donor03 | donor | Jane Eyre (C. Brontë) | 16 | Kirsten Ferreri | F | https://librivox.org/jane-eyre-by-charlotte-bront/ |
| donor04 | donor | Little Women (Alcott) | 6 | Elizabeth | F | https://librivox.org/little-women-by-louisa-may-alcott/ |

No donor shares a reader with any source.

## Music (Wikimedia Commons, marked public domain on the file pages)

Page URL = `https://commons.wikimedia.org/wiki/File:` + file name.

| Clip | Role | File on Commons | Creator / performer | Year | Clip start |
|---|---|---|---|---|---|
| src10 | source | Liszt- sonata In B Minor (piano - Jorge Bolet - 1960).mp3  | Liszt; Jorge Bolet | 1960 | 60 s |
| src11 | source | El Noi de la Mare (guitar).ogg | Jujutacular | 2009 | 5 s |
| src12 | source | Mozart - String Quartet No. 19 in C major, K465 'Dissonance' - III. Allegretto (Musopen String Quartet).flac | Mozart; Musopen String Quartet | 2012 | 30 s |
| donor05 | donor | 00 Jazz Violin Solo.ogg | VWD | 2007 | 5 s |
| donor06 | donor | Drum - Cadence A.ogg  | US Navy Band (author unknown) | n/a | 5 s |

`Flute-delay.ogg` (Chris Roddy, 2006) was downloaded but not used: it is
only 17.8 s long.

## Notes

- `data/audio.wav` and `data/sample_2.wav` were added by Member 1 for
  preprocessing tests and are not part of this dataset.
- Licence labels are as shown on the Commons and LibriVox pages at download time.
- Generated spliced files are derived from these clips. Do not redistribute
  them without checking the source licences.
