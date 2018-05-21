import pymysql
import random
import os
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


TOTAL_COUNT = 235704
TEST_COUNT = 8000
VAL_COUNT = 8000
TRAIN_COUNT = 219704

VOCAB_SIZE = 500000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

raw_dir = 'raw'
tokenized_dir = 'tokenized'
finished_files_dir = "finished_files_without_fix_missing_period"
chunks_dir = os.path.join(finished_files_dir, "chunked")
train_test_val_filenames_dir = 'train_test_val_filenames'

all_train_filenames = os.path.join(train_test_val_filenames_dir, "train_filenames")
all_test_filenames = os.path.join(train_test_val_filenames_dir, "test_filenames")
all_val_filenames  = os.path.join(train_test_val_filenames_dir, "val_filenames")

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

DB, TABLE, ID, ABSTRACT, TITLE = ['db_paper', 'paper', 'id', 'abstract', 'title']
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence


def db_to_files_with_clean():
    # db to files
    # 1. raw files: 219718 files to 'raw' dir
    # 2. three files: train_filenames, test_filenames, val_filenames
    #                 each contains file names of train_dataset, test_dataset, val_dataset respectively.
    sql = "select id, title abstract, abstract article from paper where title !='' and abstract !=''" \
    + "and title not like '%</Emphasis>%' and title not like '%</em>%' and title not like '%</Subscript>%'" \
    + "and abstract not like '%</Emphasis>%' and abstract not like '%</em>%' and abstract not like '%</Subscript>%'" \
    + "and title not like '%...'"
    print('SQL: %s' % sql)
    connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root12', db=DB, cursorclass=pymysql.cursors.DictCursor)
    cursor = connection.cursor()
    print("start to execute SQL...")
    print("This may last for several minutes...")
    cursor.execute(sql)
    result = cursor.fetchall()
    count = len(result)
    print("count: "+str(count))
    assert count == TOTAL_COUNT, "TOTAL_COUNT should be %d, but now is %d" % (TOTAL_COUNT, count)

    # write samples to files
    # format: paper_id as filename
    #         paper_abstract in first line
    #         paper_title in second line
    print("start to write raw files...")
    if not os.path.exists(raw_dir): os.makedirs(raw_dir)
    counter = 0
    for sample in result:
        if counter % 1000 == 0: print('sample[{0}] starts...'.format(counter))
        with open(os.path.join(raw_dir, sample['id']), 'w', encoding='utf-8') as writer:
            writer.write(sample['article'] + '\n')
            writer.write(sample['abstract'])
        counter += 1

    # construct train_list, test_list, val_list
    print('1.start to write train/test/val file_list...')
    train_filenames = 'train_filenames'
    test_filenames = 'test_filenames'
    val_filenames = 'val_filenames'

    index_list = []
    for i in range(count):
        index_list.append(i)
    random.shuffle(index_list)

    if not os.path.exists(train_test_val_filenames_dir): os.makedirs(train_test_val_filenames_dir)
    idx = 0
    print("test_count: {0}".format(TEST_COUNT))
    print('1.start to write test filenames...')
    with open(os.path.join(train_test_val_filenames_dir, test_filenames), 'w', encoding='utf-8') as test_writer:
        while(idx < TEST_COUNT):
            test_writer.write(result[index_list[idx]]['id'] + '\n')
            idx += 1

    print("val_count: {0}".format(VAL_COUNT))
    print('2.start to write val filenames...')
    with open(os.path.join(train_test_val_filenames_dir, val_filenames), 'w', encoding='utf-8') as val_writer:
        while(idx < TEST_COUNT+VAL_COUNT):
            val_writer.write(result[index_list[idx]]['id'] + '\n')
            idx += 1

    print('total_count: {0}, train_count: {1}'.format(count, count-TEST_COUNT-VAL_COUNT))
    print('3.start to write train filenames...')
    with open(os.path.join(train_test_val_filenames_dir, train_filenames), 'w', encoding='utf-8') as train_writer:
        while(idx < count):
            train_writer.write(result[index_list[idx]]['id'] + '\n')
            idx += 1


def chunk_file(set_name):
    in_file = 'finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    #os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def fix_missing_period(line):
	if line[-1] in END_TOKENS: return line
	return line + " ."

def get_art_abs(story_file):
    lines = []
    with open(story_file, "r", encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    # Lowercase everything
    lines = [line.lower() for line in lines]
    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    # lines = [fix_missing_period(line) for line in lines]
    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    article = fix_missing_period(line[0])
    abstract = "%s %s %s" % (SENTENCE_START, lines[1], SENTENCE_END)
    return article, abstract


def write_to_bin(file_list, out_file, makevocab=False):
    story_fnames = []
    with open(os.path.join(file_list), 'r') as f:
        for line in f:
            story_fnames.append(line.strip())
    print('file count: {0}'.format(len(story_fnames)))
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx,s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            story_file = ''
            if os.path.isfile(os.path.join(tokenized_dir, s)):
                story_file = os.path.join(tokenized_dir, s)

            # Get the strings to write to .bin file
            article, abstract = get_art_abs(story_file)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file with vocab_size: {}...".format(VOCAB_SIZE))
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    if not os.path.exists(tokenized_dir): os.makedirs(tokenized_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # db --> raw and train/test/val filename_list
    #db_to_files_with_clean()

    # raw data --> tokenized data
    # tokenize_stories(raw_dir, tokenized_dir)

    # tokenized data --> train/test/val bins
    write_to_bin(all_test_filenames, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(all_val_filenames, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(all_train_filenames, os.path.join(finished_files_dir, "train.bin"))

    # train/test/val bins  -->  chunks(train/test/val)
    chunk_all()