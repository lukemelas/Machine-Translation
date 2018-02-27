pip install spacy
python -m spacy download en
python -m spacy download de
wget "https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl"
mv multi-bleu.perl scripts/
chmod 755 scripts/multi-bleu.perl

# # Create smaller dataset for working
# ls -la .data/iwslt/de-en
# cp -r .data .data-big
# ls -la .data/iwslt/de-en
# head .data/iwslt/de-en/train.de-en.de
# head .data/iwslt/de-en/train.de-en.en
#  mv .data/iwslt/de-en/train.de-en.de .data/iwslt/de-en/train.de-en-full.de
#  mv .data/iwslt/de-en/train.de-en.en .data/iwslt/de-en/train.de-en-full.en
# head -10000 .data/iwslt/de-en/train.de-en-full.de > .data/iwslt/de-en/train.de-en.de
# head -10000 .data/iwslt/de-en/train.de-en-full.en > .data/iwslt/de-en/train.de-en.en
#  wc -l .data/iwslt/de-en/train.de-en.de
#  wc -l .data/iwslt/de-en/train.de-en.en

# # Download fasttext German embeddings
# wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec

# # Download Kaggle prediction file
# wget https://www.dropbox.com/s/o3h0sd81pjdppyj/source_test.txt

