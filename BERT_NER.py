import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse

import tensorflow as tf
from tensorflow.contrib import predictor
tf.get_logger().setLevel('ERROR')

from absl import flags,logging
from bert import tokenization

import copy
# from shutil import copyfile
from pathlib import Path
FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("label_file", None,
                    """The label file containing newline separated labels
                     of classes in data. Example: 'O'\n 'B-PER'\n etc.""")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label

    # untokenized data
    self.tokens = text.split()
    self.labels = []
    if label: self.labels = label.split()

    # tokenized data
    self.tokenized_tokens = []
    self.tokenized_labels = []


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Reads a BIO data"""
        with open(input_file, 'r', encoding='utf-8') as rf:
            lines = [];words = [];labels = []
            for line in rf:
                contents = line.strip()
                if contents.startswith("-DOCSTART-"):
                    continue
                if len(contents) == 0:  # newline
                    if len(words) == 0: continue
                    assert(len(words) == len(labels))
                    words_string = ' '.join(words)
                    labels_string = ' '.join(labels)
                    lines.append([words_string, labels_string])
                    words = []; labels = []
                    continue
                tokens = line.strip().split(' ')
                if (len(tokens) > 2):
                    print("more than 2 tokens in line _{}_".format(line.strip()))
                # assert(len(tokens) == 2) # Train data had extra spaces: "75 000"
                word  = tokens[0]
                label = tokens[-1]
                if len(tokens) == 1:
                    label = "__"
                words.append(word)
                labels.append(label)
        return lines


class NerProcessor(DataProcessor):
    def __init__(self):
        self.labels = self.get_labels()

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_examples_from_text(self, text):
        lines = []
        text_lines = text.split('\n')
        for text_line in text_lines:
            line = []
            tokens = text_line.split(' ')
            words = []
            labels = []
            for token in tokens:
                token = token.strip()
                label = "__"
                words.append(token)
                labels.append(label)
            assert(len(words) == len(labels))
            words_string = ' '.join(words)
            labels_string = ' '.join(labels)
            lines.append([words_string, labels_string])
        return self._create_example(lines, "test")


    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        Load labels from file in data directory:
        """
        data_labels = []
        with open(os.path.join(FLAGS.label_file), 'r', encoding="utf-8") as labelfile:
            data_labels = [line.strip() for line in labelfile if line.strip()]
        all_labels = ["[PAD]"] + data_labels + ["X", "[CLS]", "[SEP]"]
        return all_labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[0])
            labels = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    
    # save tokens, poss, chunks, labels back to example
    example.tokenized_tokens = tokens
    example.tokenized_labels = labels

    ntokens = []
    segment_ids = []
    label_ids = []

    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])

    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        try:
            label_ids.append(label_map[labels[i]])
        except: # "__"
            label_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding 
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(label_map["[PAD]"])
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    # if ex_index < 3:
    #     logging.info("*** Example ***")
    #     logging.info("guid: %s" % (example.guid))
    #     logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
    #     logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
    #     logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode=None):
    tf_features = []
    tf_examples = []
    for (ex_index, example) in enumerate(examples):
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_features.append(features)
        tf_examples.append(tf_example.SerializeToString())
    return tf_examples


class NamedEntity():
    """Found named entities"""

    def __init__(self, text, label=None, context='', lemmas=[], offset_start=0):
        self.context = context
        self.text = text
        self.label = label
        self.offset_start = 0
        if self.offset_start == 0 and self.context:
            self.offset_start = self.context.find(self.text)
        self.offset_end = 0
        if self.offset_start > -1:
            self.offset_end = self.offset_start + len(self.text) 

class NERDetector():

    def __init__(self, language="lv", model_dir=None, output_dir=None, saved_model_dir=None):

        if language not in ["lv", "en"]:
            raise NotImplementedError("only 'lv' and 'en' languages are supported")
        
        FLAGS([
            'BERT_NER',
            '--do_lower_case=False',
        ])
        if saved_model_dir and os.path.exists(saved_model_dir):
            FLAGS.vocab_file=os.path.join(saved_model_dir, 'vocab.txt')
            FLAGS.label_file=os.path.join(saved_model_dir, 'labels.txt')
        else:
            print("Saved model not found in {}".format(saved_model_dir))

        self.processor = NerProcessor()
        self.label_list = self.processor.labels
        self.id2label = {key: value for key, value in enumerate(self.label_list)}
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        # Load from saved_model if present, else load from estimator and export to saved_model
        try:
            subdirs = [x for x in Path(saved_model_dir).iterdir() if x.is_dir() and'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            self.predict_fn = predictor.from_saved_model(latest)
        except:
            print("Failed to load saved model {}".format(latest))

        print("NER predictor init done.")


    def get_entities(self, input_text):
        start_time = time.time()
        if input_text == "":
            return {"named entities": []}
        predict_examples = self.processor.get_examples_from_text(input_text)

        tf_feats = convert_examples_to_features(predict_examples, self.label_list,
                                                    FLAGS.max_seq_length, self.tokenizer)

        results = []
        for serialized_example in tf_feats:
            results.append(self.predict_fn({'example': [serialized_example]}))

        print("Prediction done in {} seconds".format(time.time() - start_time))
        predicted_examples = []
        for predict_example, prediction in zip(predict_examples, results):
            prediction = prediction["output"][0]

            predicted_example = copy.deepcopy(predict_example)
            predicted_example.labels = []

            tokens = predict_example.tokens
            labels = predict_example.labels
            
            tokenized_tokens = predict_example.tokenized_tokens
            tokenized_labels = predict_example.tokenized_labels
            text = predict_example.text
            length = len(tokenized_tokens)

            seq = 0
            last_label = "O"
            for token, label, p_id in zip(tokenized_tokens, tokenized_labels, prediction[1:length+1]):
                p_label = self.id2label[p_id]
                if label == 'X': continue
                if p_label == 'X': 
                    p_label = last_label
                if p_label == '[CLS]':
                    p_label = 'O'
                if 'I-' == p_label[0:2]:
                    if not p_label[2:] == last_label[2:]:
                        # Should not start with I-, replace with B-
                        p_label = 'B-'+ p_label[2:]
                last_label = p_label
                org_token = tokens[seq]
                # org_label = labels[seq]   # Oriģinālais labels __ neinteresē
                predicted_example.labels.append(p_label)
                seq += 1
            predicted_examples.append(predicted_example)

        # Extract entities from predicted_examples
        named_entities = []
        words_pos=[]
        for predicted_example in predicted_examples:
            current_entity_text = ''
            current_entity_label = ''

            for token, label in zip(predicted_example.tokens,predicted_example.labels):
                if current_entity_label and label[0] in ['B', 'O']:  
                    named_entity = NamedEntity(
                        current_entity_text, 
                        current_entity_label, 
                        context=predicted_example.text,
                        offset_start=0)
                    named_entities.append(named_entity)
                    current_entity_text = ''
                    current_entity_label = ''
                # Skip non-entities
                if label == 'O': continue
                # Append text if entity
                current_entity_text += ' '+token
                current_entity_label = label[2:]
            
        return {"named entities": [{
            "text": named_entity.text,
            "label": named_entity.label,
            } for named_entity in named_entities
        ]}
        


def main(language, model_dir, output_dir, instring, saved_model_dir):
    logging.set_verbosity(logging.INFO) # ERROR INFO

    detector = NERDetector(language, model_dir, output_dir, saved_model_dir)
    detected_entities = detector.get_entities(instring)
    print("Detected entities:\n{}".format(detected_entities))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", help="the input language")
    parser.add_argument("--model_dir", default=None, help="directory containing model files")
    parser.add_argument("--output_dir", default=None, help="directory containing fine-tuned model files")
    parser.add_argument("--instring", help="text, in which to detect entities")
    parser.add_argument("--saved_model_dir", help="dir containing saved model, bert config and labels files")

    args = parser.parse_args()
    print("Starting:")
    print(f"Language: {args.language}")
    print(f"saved_model_dir: {args.saved_model_dir}")
    print(f"instring: {args.instring}")


    main(args.language, args.model_dir, args.output_dir, args.instring, args.saved_model_dir)
