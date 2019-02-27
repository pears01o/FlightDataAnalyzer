import unittest
import sys

debug = sys.gettrace() is not None


##############################################################################
# Superclasses


class NodeTest(object):

    def test_can_operate(self):
        if not hasattr(self, 'node_class'):
            return
        kwargs = getattr(self, 'can_operate_kwargs', {})
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations(**kwargs)),
                self.operational_combination_length,
            )
        else:
            combinations = map(set, self.node_class.get_operational_combinations(**kwargs))
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)

