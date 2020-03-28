#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Shuailong
# @Email: liangshuailong@gmail.com
# @Date: 2019-05-17 15:34:30
# @Last Modified by: Shuailong
# @Last Modified time: 2019-05-17 15:34:36

# pylint: disable=no-self-use,invalid-name
import pytest
import pathlib
from allennlp.common import Params
from allennlp.common.util import ensure_list

from csqa.data.dataset_readers import CSQAReader


class TestCSQAReader:
    FIXTURES_ROOT = (pathlib.Path(__file__).parent /
                     ".." / ".." / ".." / "tests" / "fixtures").resolve()

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        params = Params({'lazy': lazy})
        reader = CSQAReader.from_params(params)
        instances = reader.read(
            str(self.FIXTURES_ROOT / 'csqa_evidence_sample.jsonl'))
        instances = ensure_list(instances)

        assert len(instances) == 5
