"""Pure channel-resolution tests for CHB-MIT montage variants."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.eeg_loader import (
    _resolve_bipolar_derivation,
    _resolve_channel,
    resolve_channel_sources,
)


class ChannelResolutionTests(unittest.TestCase):
    def test_duplicate_bipolar_channel_uses_first_copy(self):
        available = ['T8-P8-0', 'T8-P8-1', 'FZ-CZ']
        self.assertEqual(_resolve_channel('T8-P8', available), 'T8-P8-0')

    def test_common_reference_bipolar_derivation(self):
        available = ['T7-CS2', 'P7-CS2', 'T8-CS2', 'P8-CS2']
        terms, label = _resolve_bipolar_derivation('T7-P7', available)
        self.assertEqual(terms, [('T7-CS2', 1.0), ('P7-CS2', -1.0)])
        self.assertEqual(label, 'T7-CS2 - P7-CS2')

    def test_referential_electrode_bipolar_derivation(self):
        available = ['T7', 'P7', 'T8', 'P8', 'FZ', 'CZ']
        terms, label = _resolve_bipolar_derivation('FZ-CZ', available)
        self.assertEqual(terms, [('FZ', 1.0), ('CZ', -1.0)])
        self.assertEqual(label, 'FZ - CZ')

    def test_reversed_bipolar_derivation_flips_sign(self):
        terms, label = _resolve_bipolar_derivation('T7-P7', ['P7-T7'])
        self.assertEqual(terms, [('P7-T7', -1.0)])
        self.assertEqual(label, '-(P7-T7)')

    def test_chb12_zero_one_alias_is_treated_as_o1(self):
        terms, label = resolve_channel_sources('P7-O1', ['P7', '01'])
        self.assertEqual(terms, [('P7', 1.0), ('01', -1.0)])
        self.assertEqual(label, 'P7 - 01')

    def test_alias_can_resolve_direct_bipolar_spelling(self):
        self.assertEqual(_resolve_channel('P7-O1', ['P7-01']), 'P7-01')


if __name__ == '__main__':
    unittest.main()
