
#SAVED_DIR=/home/full_path/saved_model/
SAVED_DIR=./saved_model/
LANG=lv

TEST_STRING='"2001.gada 14.jūlijā Valsts policijas Latgales reģiona pārvaldes Kārtības policijas biroja Patruļpolicijas nodaļas Satiksmes uzraudzības rotas jaunākais inspektors Jānis Kalniņš sastādīja administratīvā pārkāpuma protokolu Nr.PC329 par to, ka 2005.gada 29.jūlijā  plkst.1.25 Ralfs Kalniņš vadīja vieglo automašīnu „Mercedes Benz 300”, reģistrācijas Nr.AA1010 pa Vizbuļu ielu Nr.4 virzienā no ceļa A-6 uz Parka ielu alkohola reibuma ietekmē, kuram izelpotā gaisa pārbaudē  Paula Stradiņa Klīniskajā universitātes slimnīcā konstatēta minimālā alkohola koncentrācija 10,6156 promiles, kas atbilst alkohola koncentrācijai asinīs."'

CUDA_VISIBLE_DEVICES=0 python BERT_NER.py \
	--language=$LANG \
	--saved_model_dir=$SAVED_DIR \
	--instring="$TEST_STRING"

TEST_STRING="Rīt Latvijā līs lietus, teica astrologs Jānis Kalniņš ."
#
CUDA_VISIBLE_DEVICES=0 python BERT_NER.py \
	--language=$LANG \
	--saved_model_dir=$SAVED_DIR \
	--instring="$TEST_STRING"
