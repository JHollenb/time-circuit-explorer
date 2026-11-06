import unittest
import numpy as np
from image_consequence import png_account


class ImageConsequence(unittest.TestCase):
    def test_native_noise_interface_takes_raw_not_prepacked_latents(self):
        import torch
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=None, transformer=None, is_distilled=True)
        packed, ids = pipe.prepare_latents(1, 32, 32, 32, torch.bfloat16, torch.device('cpu'), torch.Generator().manual_seed(314159))
        raw = pipe._unpack_latents_with_ids(packed, ids, 2, 2)
        self.assertEqual(tuple(raw.shape), (1, 128, 2, 2))
        self.assertEqual(tuple(packed.shape), (1, 4, 128))
        self.assertTrue(torch.equal(pipe._pack_latents(raw).view(torch.uint16), packed.view(torch.uint16)))

    def test_signed_255_difference_does_not_wrap_uint8(self):
        full = np.array([[[0, 255, 128]]], dtype=np.uint8)
        old = np.array([[[255, 0, 128]]], dtype=np.uint8)
        result = png_account(full, old, full.copy())
        self.assertEqual(result['old']['mae255'], 170.)
        self.assertEqual(result['old']['absoluteChannelErrorCounts'][255], 2)
        self.assertEqual(result['old']['absoluteChannelErrorCounts'][0], 1)
        self.assertEqual(result['pixelSquaredErrorImproved'], 1)
        self.assertEqual(result['new']['changedPixels'], 0)
