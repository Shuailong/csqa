#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-15 19:26:34
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-15 19:26:58
# Adapted from allennlp

from typing import List, Dict, Tuple
import logging
from overrides import overrides


import spacy
from spacy.language import Language as SpacyModelType
from spacy.cli.download import download as spacy_download

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_spacy_model(spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool) -> SpacyModelType:
    """
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ['vectors', 'textcat']
        if not pos_tags:
            disable.append('tagger')
        if not parse:
            disable.append('parser')
        if not ner:
            disable.append('ner')
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning(
                f"Spacy models '{spacy_model_name}' not found.  Downloading and installing.")
            spacy_download(spacy_model_name)
            # NOTE(mattg): The following four lines are a workaround suggested by Ines for spacy
            # 2.1.0, which removed the linking that was done in spacy 2.0.  importlib doesn't find
            # packages that were installed in the same python session, so the way `spacy_download`
            # works in 2.1.0 is broken for this use case.  These four lines can probably be removed
            # at some point in the future, once spacy has figured out a better way to handle this.
            # See https://github.com/explosion/spaCy/issues/3435.
            from spacy.cli import link
            from spacy.util import get_package_path
            package_path = get_package_path(spacy_model_name)
            link(spacy_model_name, spacy_model_name, model_path=package_path)
            spacy_model = spacy.load(spacy_model_name, disable=disable)

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


class SentenceSplitter():
    """
    A ``SentenceSplitter`` splits strings into sentences.
    """
    default_implementation = 'spacy'

    def split_sentences(self, text: str) -> List[str]:
        """
        Splits ``texts`` into a list of :class:`Token` objects.
        """
        raise NotImplementedError

    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        """
        This method lets you take advantage of spacy's batch processing.
        Default implementation is to just iterate over the texts and call ``split_sentences``.
        """
        return [self.split_sentences(text) for text in texts]


class SpacySentenceSplitter(SentenceSplitter):
    """
    A ``SentenceSplitter`` that uses spaCy's built-in sentence boundary detection.

    Spacy's default sentence splitter uses a dependency parse to detect sentence boundaries, so
    it is slow, but accurate.

    Another option is to use rule-based sentence boundary detection. It's fast and has a small memory footprint,
    since it uses punctuation to detect sentence boundaries. This can be activated with the `rule_based` flag.

    By default, ``SpacySentenceSplitter`` calls the default spacy boundary detector.
    """

    def __init__(self,
                 language: str = 'en_core_web_sm',
                 rule_based: bool = False) -> None:
        # we need spacy's dependency parser if we're not using rule-based sentence boundary detection.
        self.spacy = get_spacy_model(
            language, parse=not rule_based, ner=False, pos_tags=False)
        if rule_based:
            # we use `sentencizer`, a built-in spacy module for rule-based sentence boundary detection.
            # depending on the spacy version, it could be called 'sentencizer' or 'sbd'
            sbd_name = 'sbd' if spacy.__version__ < '2.1' else 'sentencizer'
            if not self.spacy.has_pipe(sbd_name):
                sbd = self.spacy.create_pipe(sbd_name)
                self.spacy.add_pipe(sbd)

    @overrides
    def split_sentences(self, text: str) -> List[str]:
        return [sent.string.strip() for sent in self.spacy(text).sents]

    @overrides
    def batch_split_sentences(self, texts: List[str]) -> List[List[str]]:
        return [[sentence.string.strip() for sentence in doc.sents] for doc in self.spacy.pipe(texts)]
