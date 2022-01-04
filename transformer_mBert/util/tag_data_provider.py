import json
import torch
import torch.utils.data as data
import numpy as np

from basic.util import getVideoId
from util.vocab import clean_str_cased

from transformers import BertTokenizer

VIDEO_MAX_LEN = 64


def convert_example_to_features(tokens, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    def convert_tokens_to_ids(tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(tokenizer.vocab.get(token, tokenizer.vocab['[UNK]']))

            # ids.append(tokenizer.vocab[token])
        return ids

    assert isinstance(tokens, list)

    input_tokens = []
    val_pos = []
    end_idx = 0
    for word in tokens:
        b_token = tokenizer.tokenize(word)  # we expect |token| = 1
        # b_token = word
        input_tokens.extend(b_token)
        val_pos.append(end_idx)
        end_idx += len(b_token)

    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = convert_tokens_to_ids(input_tokens)

    assert len(input_ids) == len(input_tokens)
    assert max(input_ids) < len(tokenizer.vocab)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    return {
        'input_len': len(tokens),
        'input_tokens': input_tokens,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'out_positions': val_pos
    }


def create_tokenizer():
    model_name_or_path = 'bert-base-multilingual-cased'
    # model_name_or_path = 'bert-base-cased'
    # model_name_or_path = 'bert-large-uncased'
    do_lower_case = True
    cache_dir = 'data/cache_dir'
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=cache_dir)
    return tokenizer


def tokenize_caption(tokenizer, raw_caption, cap_id, special_tokens=True, type='EN'):
    # print(type, '--------')

    if (type == 'EN'):
        word_list = clean_str_cased(raw_caption)
        text_caption = word_list
        #txt_caption = " ".join(word_list)
        # Remove whitespace at beginning and end of the sentence.
        #txt_caption = txt_caption.strip()
        # Add period at the end of the sentence if not already there.
        try:
            if text_caption[-1] not in [".", "?", "!"]:
                text_caption += "."
        except:
            print(raw_caption)
            print(text_caption)
            print(cap_id)
        # txt_caption = txt_caption.capitalize()
        # tokens = tokenizer.tokenize(txt_caption)
        # if special_tokens:
        #     cls = [tokenizer.cls_token]
        #     sep = [tokenizer.sep_token]  # [SEP] token
        # tokens = cls + tokens + sep
        # # tokens = tokens[:self.max_text_words]
        # # Make sure that the last token is
        # # the [SEP] token
        # if special_tokens:
        #     tokens[-1] = tokenizer.sep_token
        #
        # ids = tokenizer.convert_tokens_to_ids(tokens)

        # ids = tokenizer.encode(txt_caption, add_special_tokens=True)
        # BertFeats_of_token = convert_example_to_features(txt_caption, tokenizer)
    # else:
    # ids = tokenizer.encode(raw_caption, add_special_tokens=True)
    # BertFeats_of_token = convert_example_to_features(raw_caption, tokenizer)

    return text_caption


def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids, vid_tag = zip(*data)
    # BERT
    videos, bert_cap, idxs, cap_ids, video_ids, vid_tag = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0
    videos_tag = torch.stack(vid_tag, 0)

    if bert_cap[0] is not None:
        batch_size = len(bert_cap)
        bert_inst_batch = []
        lengths = [len(cap) for cap in bert_cap]
        words_mask = torch.zeros(batch_size, max(lengths))
        for i, cap in enumerate(bert_cap):
            end = lengths[i]
            words_mask[i, :end] = 1.0
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        for sen in bert_cap:
            bert_inst_batch.append(convert_example_to_features(sen, tokenizer))
        bert_max_length = max([len(inst['input_ids']) for inst in bert_inst_batch])
        batch_length = max(lengths)
        bert_inputs_ids = np.zeros([batch_size, bert_max_length], dtype=np.int64)
        bert_input_mask = np.zeros([batch_size, bert_max_length], dtype=np.int64)
        bert_out_positions = np.empty([batch_size, batch_length], dtype=np.int64)
        for i in range(batch_size):
            berts = bert_inst_batch[i]
            bert_inputs_ids[i, :len(berts['input_ids'])] = berts['input_ids']
            bert_input_mask[i, :len(berts['input_mask'])] = berts['input_mask']
            required_pad = batch_length - len(berts['out_positions'])
            if required_pad > 0:
                low = berts['out_positions'][-1]
                assert (bert_max_length - 2) > low
                bert_out_positions[i] = berts['out_positions'] + [low] * required_pad
            else:
                bert_out_positions[i] = berts['out_positions']
        
    else:
        bert_inputs_ids = None
        bert_input_mask = None
        bert_out_positions = None
        bert_max_length = None
        batch_length = None
        lengths = None
        words_mask = None

    lengths = torch.IntTensor(lengths)
    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    # BERT
    # text_data = (bert_target, lengths, words_mask)
    text_data = ((bert_inputs_ids, bert_input_mask, bert_out_positions,
                 bert_max_length, batch_length), lengths, words_mask)

    return video_data, text_data, videos_tag, idxs, cap_ids, video_ids


def collate_frame(data):
    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions, cap_bows, idxs, cap_ids = zip(*data)
    bert_cap, idxs, cap_ids = zip(*data)

    # BERT
    if bert_cap[0] is not None:
        batch_size = len(bert_cap)
        lengths = [len(cap) for cap in bert_cap]
        words_mask = torch.zeros(batch_size, max(lengths))
        for i, cap in enumerate(bert_cap):
            end = lengths[i]
            words_mask[i, :end] = 1.0

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        bert_inst_batch = []
        for sen in bert_cap:
            bert_inst_batch.append(convert_example_to_features(sen, tokenizer))
        bert_max_length = max([len(inst['input_ids']) for inst in bert_inst_batch])
        batch_length = max(lengths)
        bert_inputs_ids = np.zeros([batch_size, bert_max_length], dtype=np.int64)
        bert_input_mask = np.zeros([batch_size, bert_max_length], dtype=np.int64)
        bert_out_positions = np.empty([batch_size, batch_length], dtype=np.int64)
        for i in range(batch_size):
            berts = bert_inst_batch[i]
            bert_inputs_ids[i, :len(berts['input_ids'])] = berts['input_ids']
            bert_input_mask[i, :len(berts['input_mask'])] = berts['input_mask']
            required_pad = batch_length - len(berts['out_positions'])
            if required_pad > 0:
                low = berts['out_positions'][-1]
                assert (bert_max_length - 2) > low
                bert_out_positions[i] = berts['out_positions'] + [low] * required_pad
            else:
                bert_out_positions[i] = berts['out_positions']
    else:
        bert_inputs_ids = None
        bert_input_mask = None
        bert_out_positions = None
        bert_max_length = None
        batch_length = None
        lengths = None
        words_mask = None

    lengths = torch.IntTensor(lengths)
    # BERT
    # text_data = (bert_target, lengths, words_mask)
    text_data = ((bert_inputs_ids, bert_input_mask, bert_out_positions,
                 bert_max_length, batch_length), lengths, words_mask)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, tag_path, tag_vocab_path, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        self.tag_path = tag_path
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                #caption, trans_caption = caption.split("||")

                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
        self.length = len(self.cap_ids)

        self.tag_vocab_list = json.load(open(tag_vocab_path, 'r'))
        self.tag_vocab_size = len(self.tag_vocab_list)
        self.tag2idx = dict(zip(self.tag_vocab_list, range(self.tag_vocab_size)))

        # self.vid2tags = json.load(open(tag_path, 'r'))    #read the json file of tag
        self.vid2tags = {}
        if tag_path is not None:
            for line in open(tag_path).readlines():
                # print(line)
                if len(line.strip().split("\t", 1)) < 2:  # no tag available for a specific video
                    vid = line.strip().split("\t", 1)[0]
                    self.vid2tags[vid] = []
                else:
                    vid, or_tags = line.strip().split("\t", 1)
                    tags = [x.split(':')[0] for x in or_tags.strip().split()]

                    # weighed concept scores
                    scores = [float(x.split(':')[1]) for x in or_tags.strip().split()]
                    scores = np.array(scores) / max(scores)

                    self.vid2tags[vid] = list(zip(tags, scores))
        # BERT
        self.tokenizer = create_tokenizer()

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        if self.tag_path is not None:
            vid_tag_str = self.vid2tags[video_id]  # string representation
            tag_in_vocab = [tag_score for tag_score in vid_tag_str if tag_score[0] in self.tag2idx]
            tag_list = [self.tag2idx[tag_score[0]] for tag_score in tag_in_vocab]  # index representation
            score_list = [tag_score[1] for tag_score in tag_in_vocab]
            tag_one_hot = torch.zeros(
                self.tag_vocab_size)  # build zero vector of tag vocabulary that is used to represent tags by one-hot
            for idx, tag_idx in enumerate(tag_list):
                tag_one_hot[tag_idx] = score_list[idx]  # one-hot
        else:
            tag_one_hot = torch.zeros(self.tag_vocab_size)
        vid_tag = torch.Tensor(np.array(tag_one_hot))

        # BERT
        caption = self.captions[cap_id]
        bertfeatures_token = tokenize_caption(self.tokenizer, caption, cap_id)
        # bert_tensor = torch.Tensor(bert_ids)

        # BERT
        return frames_tensor, bertfeatures_token, index, cap_id, video_id, vid_tag

    def __len__(self):
        return self.length


class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """

    def __init__(self, visual_feat, video2frames=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.length = len(self.cap_ids)
        # BERT
        self.tokenizer = create_tokenizer()

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        # BERT
        caption = self.captions[cap_id]
        bertfeatures_token = tokenize_caption(self.tokenizer, caption, cap_id)
        # bert_tensor = torch.Tensor(bert_ids)

        return bertfeatures_token, index, cap_id

    def __len__(self):
        return self.length


def get_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, batch_size=100, num_workers=2,
                     video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path, bow2vec,
                                          vocab, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], None, tag_vocab_path, bow2vec, vocab,
                                        video2frames=video2frames['val'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=(x == 'train'),
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn)
                    for x in cap_files}
    return data_loaders


def get_train_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, batch_size=100, num_workers=2,
                           video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path,
                                          video2frames=video2frames['train'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=(x == 'train'),
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn)
                    for x in cap_files if x == 'train'}
    return data_loaders


def get_test_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, batch_size=100, num_workers=2,
                          video2frames=None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], tag_path['test'], tag_vocab_path,
                                         video2frames=video2frames['test'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_frame_gru_fn)
                    for x in cap_files}
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None, video_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames, video_ids=video_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(cap_file, batch_size=100, num_workers=2):
    dset = TxtDataSet4DualEncoding(cap_file)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader


if __name__ == '__main__':
    pass
