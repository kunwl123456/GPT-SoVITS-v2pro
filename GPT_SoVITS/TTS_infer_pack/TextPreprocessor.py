import os
import sys
import threading

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter,LangSegment
from text import chinese
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from tools.i18n.i18n import I18nAuto, scan_language_list
from line_profiler import profile

language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(["!", "?", "…", ",", ".", "-"])


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: str, threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    # language=language.replace("all_","")
    # if language == "zh":
    #     bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    # else:
    #     bert = torch.zeros(
    #         (1024, len(phones)),
    #         dtype=torch.float16 if is_half == True else torch.float32,
    #     ).to(device)
    bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    # bert = torch.zeros(
    #     (1024, len(phones)),
    #     dtype=torch.float16 if is_half == True else torch.float32,
    # ).to(device)
    return bert
import itertools

class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()

    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        print(f"############ {i18n('切分文本')} ############")
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(f"############ {i18n('提取文本Bert特征')} ############")
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    # def pre_seg_text(self, text: str, lang: str, text_split_method: str):
    #     text = text.strip("\n")
    #     if len(text) == 0:
    #         return []
    #     if text[0] not in splits and len(get_first(text)) < 4:
    #         text = "。" + text if lang != "en" else "." + text
    #     print(i18n("实际输入的目标文本:"))
    #     print(text)

    #     seg_method = get_seg_method(text_split_method)
    #     text = seg_method(text)

    #     while "\n\n" in text:
    #         text = text.replace("\n\n", "\n")

    #     _texts = text.split("\n")
    #     _texts = self.filter_text(_texts)
    #     _texts = merge_short_text_in_array(_texts, 5)
    #     texts = []

    #     for text in _texts:
    #         # 解决输入目标文本的空行导致报错的问题
    #         if len(text.strip()) == 0:
    #             continue
    #         if not re.sub("\W+", "", text):
    #             # 检测一下，如果是纯符号，就跳过。
    #             continue
    #         if text[-1] not in splits:
    #             text += "。" if lang != "en" else "."

    #         # 解决句子过长导致Bert报错的问题
    #         if len(text) > 510:
    #             texts.extend(split_big_text(text))
    #         else:
    #             texts.append(text)

    #     print(i18n("实际输入的目标文本(切句后):"))
    #     print(texts)
    #     return texts
    def pre_seg_text(self, texts: List[str], lang: str, text_split_method: str):
        result = []
        for text in texts:
            text = text.strip("\n")
            if text[0] not in splits and len(get_first(text)) < 4:
                text = "。" + text if lang != "en" else "." + text
            print(i18n("实际输入的目标文本:"))
            print(text)

            seg_method = get_seg_method(text_split_method)
            text = seg_method(text)

            while "\n\n" in text:
                text = text.replace("\n\n", "\n")

            _texts = text.split("\n")
            _texts = self.process_text(_texts)
            _texts = merge_short_text_in_array(_texts, 5)
            batch_texts = []

            for t in _texts:
                # 解决输入目标文本的空行导致报错的问题
                if len(t.strip()) == 0:
                    continue
                if not re.sub("\W+", "", t):
                    # 检测一下，如果是纯符号，就跳过。
                    continue
                if t[-1] not in splits:
                    t += "。" if lang != "en" else "."

                # 解决句子过长导致Bert报错的问题
                if len(t) > 510:
                    batch_texts.extend(split_big_text(t))
                else:
                    batch_texts.append(t)

            print(i18n("实际输入的目标文本(切句后):"))
            print(batch_texts)
            result.append(batch_texts)
        
        return result
    


    # def segment_and_extract_feature_for_text(
    #     self, text: str, language: str, version: str = "v1"
    # ) -> Tuple[list, torch.Tensor, str]:
    #     #return self.get_phones_and_bert(text, language, version)
    #     return self.get_phones_and_bert(text, language)

    # def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False):
    #     with self.bert_lock:
    #         text = re.sub(r' {2,}', ' ', text)
    #         textlist = []
    #         langlist = []
    #         if language == "all_zh":
    #             for tmp in LangSegmenter.getTexts(text,"zh"):
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         elif language == "all_yue":
    #             for tmp in LangSegmenter.getTexts(text,"zh"):
    #                 if tmp["lang"] == "zh":
    #                     tmp["lang"] = "yue"
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         elif language == "all_ja":
    #             for tmp in LangSegmenter.getTexts(text,"ja"):
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         elif language == "all_ko":
    #             for tmp in LangSegmenter.getTexts(text,"ko"):
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         elif language == "en":
    #             langlist.append("en")
    #             textlist.append(text)
    #         elif language == "auto":
    #             for tmp in LangSegmenter.getTexts(text):
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         elif language == "auto_yue":
    #             for tmp in LangSegmenter.getTexts(text):
    #                 if tmp["lang"] == "zh":
    #                     tmp["lang"] = "yue"
    #                 langlist.append(tmp["lang"])
    #                 textlist.append(tmp["text"])
    #         else:
    #             for tmp in LangSegmenter.getTexts(text):
    #                 if langlist:
    #                     if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
    #                         textlist[-1] += tmp["text"]
    #                         continue
    #                 if tmp["lang"] == "en":
    #                     langlist.append(tmp["lang"])
    #                 else:
    #                     # 因无法区别中日韩文汉字,以用户输入为准
    #                     langlist.append(language)
    #                 textlist.append(tmp["text"])
    #         # print(textlist)
    #         # print(langlist)
    #         phones_list = []
    #         bert_list = []
    #         norm_text_list = []
    #         for i in range(len(textlist)):
    #             lang = langlist[i]
    #             phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
    #             bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
    #             phones_list.append(phones)
    #             norm_text_list.append(norm_text)
    #             bert_list.append(bert)
    #         bert = torch.cat(bert_list, dim=1)
    #         phones = sum(phones_list, [])
    #         norm_text = "".join(norm_text_list)

    #         if not final and len(phones) < 6:
    #             return self.get_phones_and_bert("." + text, language, version, final=True)

    #         return phones, bert, norm_text

    def get_phones_and_bert(self,text,language):
        def _flatten_str_list(x):
            for e in x if isinstance(x, (list, tuple)) else [x]:
                if isinstance(e, (list, tuple)):
                    yield from _flatten_str_list(e)
                elif isinstance(e, str) and e.strip():
                    yield e
        if language in {"en","all_zh","all_ja"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert = get_bert_feature(norm_text, word2ph).to(device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if is_half == True else torch.float32,
                ).to(device)
        elif language in {"zh", "ja","auto"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            for s in _flatten_str_list(text):
                if language == "auto":
                    for tmp in LangSegment.getTexts(s):
                        if tmp["lang"] == "ko":
                            langlist.append("zh")
                            textlist.append(tmp["text"])
                        else:
                            langlist.append(tmp["lang"])
                            textlist.append(tmp["text"])
                else:
                    for tmp in LangSegment.getTexts(s):
                        if tmp["lang"] == "en":
                            langlist.append(tmp["lang"])
                        else:
                            # 因无法区别中日文汉字,以用户输入为准
                            langlist.append(language)
                        textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

            return [phones],[bert.to(dtype)],[norm_text]

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    # def clean_text_inf(self, text: str, language: str, version: str = "v2"):
    #     language = language.replace("all_", "")
    #     phones, word2ph, norm_text = clean_text(text, language, version)
    #     phones = cleaned_text_to_sequence(phones, version)
    #     return phones, word2ph, norm_text

    # def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
    #     language = language[0].replace("all_", "")
    #     if language == "zh":
    #         feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
    #     else:
    #         feature = torch.zeros(
    #             (1024, len(phones)),
    #             dtype=torch.float32,
    #         ).to(self.device)

    #     return feature


    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = "".join(re.escape(p) for p in punctuation)
        pattern = f"([{punctuations}])([{punctuations}])+"
        result = re.sub(pattern, r"\1", text)
        return result

##       add   kuaanglixiang     2025.10.16

    def process_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text
    def clean_text_inf(self,text:str, language:str):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text
    
    
    def segment_and_extract_feature_for_text(
        self, texts: list[list[str]], language: str, is_prompt: bool = False
    ) -> Tuple[list, torch.Tensor, str, list[int]]:
        # 传入不展平
        textlist, langlist, data_idx, language_data_idx = self.get_text_language(texts, language)
        # 传出idx以及展平
        if len(textlist) == 0:
            return None, None, None

        phones, bert_features, norm_text, data_idx = self.extract_bert_feature(textlist, langlist, data_idx, language_data_idx, is_prompt)
        return phones, bert_features, norm_text, data_idx
    

    def get_text_language(self, requests: list[list[str]], language: str) -> Tuple[list, list, list, list]:
        """
        返回的两个data_idx分别对应：
        1. 句子属于哪个用户的
        2. 按语言分词前属于哪个句子
        
        Example:
        input:
        >>> requests = [["你好This is a test", "第二段分句"], ["这是别的用户"]]
        
        output:
        >>> textlist_res = ["你好", "This is a test", "第二段分句", "这是别的用户"]
        >>> langlist_res = ["zh", "en", "zh", "zh"]
        >>> data_idx = [0, 0, 0, 1]
        >>> language_data_idx = [[0, 0, 1], [0]]
        
        """
        textlist_res = []
        langlist_res = []
        language_data_idx = []
        # requests = [["你好This is a test", "第二段分句"], ["这是别的用户"]]
        for request in requests:
            # request = ["你好This is a test", "第二段分句"]
            textlist_1sentence = []
            langlist_1sentence = []
            data_idx = []
            for i, text in enumerate(request):
                # text = "你好This is a test"
                textlist = []
                langlist = []
                LangSegment.setfilters(["zh", "ja", "en", "ko"])
                if language == "auto":
                    for tmp in LangSegment.getTexts(text):
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
                        data_idx.append(i)
                elif language == "auto_yue":
                    for tmp in LangSegment.getTexts(text):
                        if tmp["lang"] == "zh":
                            tmp["lang"] = "yue"
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
                        data_idx.append(i)
                else:
                    for tmp in LangSegment.getTexts(text):
                        if tmp["lang"] == "en":
                            langlist.append(tmp["lang"])
                        else:
                            # 因无法区别中日韩文汉字,以用户输入为准
                            langlist.append(language)
                        textlist.append(tmp["text"])
                        data_idx.append(i)
                textlist_1sentence.extend(textlist)
                langlist_1sentence.extend(langlist)
            # data_idx = [0, 0, 1]
            textlist_res.append(textlist_1sentence)
            langlist_res.append(langlist_1sentence)
            language_data_idx.append(data_idx)
        request_data_idx = []
        for i, request in enumerate(textlist_res):
            request_data_idx.extend([i] * len(request))
        return sum(textlist_res, []), sum(langlist_res, []), request_data_idx, language_data_idx
    
    @profile
    def extract_bert_feature(self, textlist: list, langlist: list, data_idx: list, language_data_idx: list, is_prompt: bool = False):
        phones_list = []
        bert_list = []
        norm_text_list = []
        word2ph_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            word2ph_list.append(word2ph)
        bert_list = self.get_bert_inf(phones_list, word2ph_list, norm_text_list, langlist)
            # bert_list.append(bert)
        bert_res = []
        phones_res = []
        norm_text_res = []
        new_data_idx = []
        idx = 0
        if is_prompt:
            for i in range(data_idx[-1]+1):
                bert = torch.cat(bert_list[idx:idx+data_idx.count(i)], dim=1).to(torch.float32)
                phones = sum(phones_list[idx:idx+data_idx.count(i)], [])
                norm_text = "".join(norm_text_list[idx:idx+data_idx.count(i)])
                idx += data_idx.count(i)
                bert_res.append(bert)
                phones_res.append(phones)
                norm_text_res.append(norm_text)
        else:
            # phones_list = [["你好", "This is a test", "第二段分句"],["这是别的用户"]]
            phones_list = self.divide_batch(phones_list, data_idx)
            bert_list = self.divide_batch(bert_list, data_idx)
            norm_text_list = self.divide_batch(norm_text_list, data_idx)
            langlist = self.divide_batch(langlist, data_idx)
            # 有几个请求就循环几次
            for i, (phones, bert, norm_text, lang) in enumerate(zip(phones_list, bert_list, norm_text_list, language_data_idx)):
                # 根据lang合并连续相同语言的文本
                merged = []
                for key, group in itertools.groupby(zip(phones, bert, norm_text, lang), key=lambda x: x[3]):
                    group_list = list(group)
                    merged_phones = list(itertools.chain.from_iterable(item[0] for item in group_list))
                    merged_bert = torch.cat([item[1] for item in group_list], dim=1)
                    merged_text = ''.join(item[2] for item in group_list)
                    merged.append((merged_phones, merged_bert, merged_text))

                phones, bert, norm_text = zip(*merged)
                phones_res += list(phones)
                bert_res += list(bert)
                norm_text_res += norm_text
                new_data_idx.extend([i] * len(phones))
                
        if new_data_idx:
            data_idx = new_data_idx
        return phones_res, bert_res, norm_text_res, data_idx
    
    def get_bert_inf(
        self, phones: list, word2ph: list, norm_text: list, languages: list
    ):  # TODO: 测试全语言获取bert,bert换成BGE-M3.
        languages = [language.replace("all_", "") for language in languages]
        features = []
        zh_norm_text = []
        zh_word2ph = []
        for i in range(len(languages)):
            if languages[i] == "zh":
                zh_norm_text.append(norm_text[i])
                zh_word2ph.append(word2ph[i])
                features.append(None)
            else:
                feature = torch.zeros(
                    (1024, len(phones[i])),
                    dtype=torch.float32,
                ).to(self.device)
                features.append(feature)
        if len(zh_norm_text) > 0:
            # 修复：循环调用get_bert_feature处理每个文本，而不是传入列表
            zh_bert = []
            for text, word2ph in zip(zh_norm_text, zh_word2ph):
                bert_feat = self.get_bert_feature(text, word2ph)
                zh_bert.append(bert_feat)
        zh_idx = 0
        for i in range(len(features)):
            if features[i] is None:
                features[i] = zh_bert[zh_idx]
                zh_idx += 1
        return features
    
    def divide_batch(self, data: list | torch.LongTensor, idx: list):
        """
        Divide the batch into multiple groups according to the idx.

        Args:
            data (list): the data to be divided. [a,b,c,d,e]
            idx (list): the index of the data.   [0,0,1,2,2]

        Returns:
            list: the divided data. [[a, b], [c], [d, e]]
        """
        if isinstance(data, torch.LongTensor):
            data = data.tolist()
        grouped_results = {}
        for item, group_index in zip(data, idx):
            if group_index not in grouped_results:
                grouped_results[group_index] = []
            grouped_results[group_index].append(item)
        return list(grouped_results.values())