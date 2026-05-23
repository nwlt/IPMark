import spacy
import hashlib
import hmac
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

POS_MAP = {
    "NOUN": wn.NOUN,
    "PROPN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}


def get_hypernyms_for_word(word, pos_tag=None, top_n=1):
    if not word:
        return ""

    synsets = wn.synsets(word, pos=pos_tag) if pos_tag else wn.synsets(word)
    if not synsets:
        synsets = wn.synsets(word)
        if not synsets:
            return word

    s = synsets[0]
    hypers = s.hypernyms()
    if not hypers:
        return word

    names = []
    for h in hypers[:top_n]:
        lemma_names = h.lemma_names()
        if lemma_names:
            names.append(lemma_names[0].replace("_", " "))
    if names:
        return ",".join(names)
    else:
        return word


def generate_watermark_keys(text, model_id="1234", user_id="654321", base_key="naninifa545a454f7ae7"):
    doc = nlp(text)

    POS_TO_CHUNK = {
        "NOUN": "NP",
        "PROPN": "NP",
        "PRON": "NP",
        "VERB": "VP",
        "AUX": "VP",
        "ADJ": "ADJP",
        "ADV": "ADVP",
        "ADP": "PP",
        "DET": "NP",
        "CCONJ": "CONJ",
        "SCONJ": "SCONJ",
    }

    def prune_tree_keep_structure(token):
        children_pruned = [prune_tree_keep_structure(child) for child in token.children]
        label = POS_TO_CHUNK.get(token.pos_, token.dep_)

        if not children_pruned:
            return f"[{label}]"

        return f"[{label}{''.join(children_pruned)}]"

    def get_syntax_structures(doc):
        root_tokens = [token for token in doc if token.dep_ == "ROOT"]
        if not root_tokens:
            return ""
        root = root_tokens[0]
        pruned_tree = prune_tree_keep_structure(root)
        return pruned_tree

    component_syntax = get_syntax_structures(doc)

    subj_token = None
    verb_token = None
    obj_token = None

    for token in doc:
        if token.dep_ == "ROOT":
            verb_token = token
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subj_token = child
                elif child.dep_ in ["attr", "acomp", "dobj"]:
                    obj_token = child
                elif child.dep_ == "xcomp":
                    for xcomp_child in child.children:
                        if xcomp_child.dep_ == "dobj":
                            obj_token = xcomp_child
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj_token = pobj

    subject_raw = subj_token.text if subj_token is not None else ""
    verb_raw = verb_token.text if verb_token is not None else ""
    obj_raw = obj_token.text if obj_token is not None else ""

    subject_lemma = subj_token.lemma_ if subj_token is not None else subject_raw
    verb_lemma = verb_token.lemma_ if verb_token is not None else verb_raw
    obj_lemma = obj_token.lemma_ if obj_token is not None else obj_raw

    subj_pos = POS_MAP.get(subj_token.pos_, None) if subj_token is not None else None
    verb_pos = POS_MAP.get(verb_token.pos_, None) if verb_token is not None else None
    obj_pos = POS_MAP.get(obj_token.pos_, None) if obj_token is not None else None

    subject_hyper = get_hypernyms_for_word(subject_lemma, pos_tag=subj_pos) if subject_lemma else ""
    verb_hyper = get_hypernyms_for_word(verb_lemma, pos_tag=verb_pos) if verb_lemma else ""
    obj_hyper = get_hypernyms_for_word(obj_lemma, pos_tag=obj_pos) if obj_lemma else ""

    semanteme = subject_hyper + verb_hyper + obj_hyper

    component_hash = hashlib.sha256((semanteme + base_key).encode()).hexdigest()
    last_int = int(component_hash[-1], 16)

    if last_int < 8:
        key_used = model_id.encode()
        key_type = "model_id"
    else:
        key_used = (model_id + "." + user_id).encode()
        key_type = "model_id.user_id"

    hmac_syntax = hmac.new(key_used, component_syntax.encode(), hashlib.sha256).hexdigest()
    hmac_semanteme = hmac.new(key_used, semanteme.encode(), hashlib.sha256).hexdigest()

    last_char_hmac_syntax = hmac_syntax[-1]
    last_int__hmac_syntax = int(last_char_hmac_syntax, 16)
    last_char_hmac_semanteme = hmac_semanteme[-1]
    last_int__hmac_semanteme = int(last_char_hmac_semanteme, 16)

    return key_type, last_int__hmac_syntax, last_int__hmac_semanteme
