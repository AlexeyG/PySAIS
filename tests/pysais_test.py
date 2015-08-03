#! /usr/bin/env python
# -*- coding:utf-8 -*-

import pysais
import unittest


class TestPySAIS(unittest.TestCase):

    def setUp(self):
        None

    def test_mississippi(self):
        t = 'mississippi'
        sa = pysais.sais(t)
        lcp, lcp_lm, lcp_mr = pysais.lcp(t, sa)
        self.assertEquals([10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2],
                          [x for x in sa])
        self.assertEquals([1, 1, 4, 0, 0, 1, 0, 2, 1, 3, 0],
                          [x for x in lcp])


if __name__ == '__main__':
    unittest.main()
