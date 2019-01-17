#! /usr/bin/env python
# -*- coding:utf-8 -*-

import pysais
import unittest
import numpy


class TestPySAIS(unittest.TestCase):

    def setUp(self):
        None

    def test_mississippi(self):

        # with strings
        t = 'mississippi'
        sa = pysais.sais(t)
        lcp, lcp_lm, lcp_mr = pysais.lcp(t, sa)

        # with ints
        # the following array encodes mississipi with the following sorted
        # character mapping: i=0, m=1, p=2, s=3
        # the following array
        t_int = numpy.array([1, 0, 3, 3, 0, 3, 3, 0, 2, 2, 0],
                            dtype=numpy.int32)
        sa_int = pysais.sais_int(t_int, 4)
        lcp_int, lcp_lm_int, lcp_mr_int = pysais.lcp_int(t_int, sa_int)

        # the expected outputs are the same in both cases
        expected_sa = [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]
        expected_lcp = [1, 1, 4, 0, 0, 1, 0, 2, 1, 3, 0]

        self.assertEquals(expected_sa, [x for x in sa])
        self.assertEquals(expected_sa, [x for x in sa_int])
        self.assertEquals(expected_lcp, [x for x in lcp])
        self.assertEquals(expected_lcp, [x for x in lcp_int])


if __name__ == '__main__':
    unittest.main()
