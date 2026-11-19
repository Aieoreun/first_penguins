import unittest

import numpy as np

from core.kb import KBChunk
from core.search import Searcher
from pipelines.kimjaehak.section_ranker import SectionFeatures


class SectionFeatureTests(unittest.TestCase):
    def chunks(self):
        return [KBChunk('PY-01-01','python','파이썬','','PY-01-01 파이썬\n변수 apple = 4와 pear = 2의 합은 6입니다.'),
                KBChunk('FIN-01-01','finance','저축','','FIN-01-01 저축\n예금 이자는 3퍼센트입니다.')]

    def test_relevant_section_has_stronger_matching_features(self):
        f=SectionFeatures(Searcher(self.chunks()))
        x=f.transform([{'role':'user','content':'파이썬 apple pear 합은?'}])
        self.assertEqual(x.shape,(2,24))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(x[0,0],x[1,0])
        self.assertGreater(x[0,6],x[1,6])

    def test_section_order_cannot_change_per_section_features(self):
        a=SectionFeatures(Searcher(self.chunks()))
        b=SectionFeatures(Searcher(list(reversed(self.chunks()))))
        messages=[{'role':'user','content':'예금 이자 3퍼센트'}]
        ax,bx=a.transform(messages),b.transform(messages)
        for sid in a.ids:np.testing.assert_allclose(ax[a.ids.index(sid)],bx[b.ids.index(sid)])

    def test_empty_question_is_finite_and_preserves_equal_scores(self):
        x=SectionFeatures(Searcher(self.chunks())).transform([])
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[0],x[1])
