import unittest

from open_kernels.gemma4_text_plan import (Gemma4PlanError, from_hf_config,
                                            projections, projection_geometries,
                                            projection_image_geometry, Projection)
from open_kernels.build_gemma4_text_projections import image_plan


def config(variant):
    profiles = {
        "E2B": ("gemma4_text", 1536, 35, 6144, 8, 1, None, 512, 20, 256, True, False, 131072, 5),
        "E4B": ("gemma4_text", 2560, 42, 10240, 8, 2, None, 512, 18, 256, False, False, 131072, 6),
        "12B": ("gemma4_unified_text", 3840, 48, 15360, 16, 8, 1, 1024, 0, 0, False, True, 262144, 6),
    }
    mt, hidden, depth, ff, heads, kvh, global_kvh, window, shared, ple, double, k_eq_v, ctx, period = profiles[variant]
    return {"model_type": mt, "hidden_size": hidden, "num_hidden_layers": depth,
            "intermediate_size": ff, "num_attention_heads": heads,
            "num_key_value_heads": kvh, "num_global_key_value_heads": global_kvh,
            "head_dim": 256, "global_head_dim": 512, "sliding_window": window,
            "num_kv_shared_layers": shared, "hidden_size_per_layer_input": ple,
            "use_double_wide_mlp": double, "attention_k_eq_v": k_eq_v,
            "max_position_embeddings": ctx, "vocab_size": 262144,
            "hidden_activation": "gelu_pytorch_tanh", "enable_moe_block": False,
            "attention_bias": False, "final_logit_softcapping": 30.0,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25},
            },
            "layer_types": ["full_attention" if (i + 1) % period == 0 else "sliding_attention"
                            for i in range(depth)]}


class Gemma4TextPlanTest(unittest.TestCase):
    def test_e2b_shares_by_attention_type_and_doubles_only_tail_mlp(self):
        plan = from_hf_config({"model_type": "gemma4", "text_config": config("E2B")})
        self.assertEqual(len(plan.layers), 35)
        self.assertEqual((plan.layers[13].stores_shared_kv, plan.layers[14].stores_shared_kv), (True, True))
        self.assertEqual([x.kv_source_layer for x in plan.layers[15:20]], [13, 13, 13, 13, 14])
        self.assertEqual([x.mlp_width for x in plan.layers[14:16]], [6144, 12288])
        self.assertEqual((plan.layers[4].query_width, plan.layers[4].kv_width, plan.layers[4].rotary_dim),
                         (4096, 512, 128))

    def test_e4b_sharing_preserves_ordinary_mlp_width(self):
        plan = from_hf_config(config("E4B"))
        self.assertEqual([x.kv_source_layer for x in plan.layers[24:30]], [22, 22, 22, 22, 22, 23])
        self.assertEqual({x.mlp_width for x in plan.layers}, {10240})
        self.assertEqual((plan.layers[5].query_width, plan.layers[5].kv_width), (4096, 1024))

    def test_12b_full_reuses_k_projection_as_v(self):
        plan = from_hf_config({"model_type": "gemma4_unified", "text_config": config("12B")})
        self.assertEqual((plan.layers[5].kv_heads, plan.layers[5].kv_width), (1, 512))
        self.assertTrue(plan.layers[5].alternative_k_eq_v)
        self.assertFalse(plan.layers[5].has_v_projection)
        self.assertTrue(plan.layers[4].has_v_projection)
        self.assertEqual({x.kv_source_layer for x in plan.layers}, set(range(48)))

    def test_refuses_geometry_or_schedule_drift(self):
        cfg = config("E2B")
        cfg["layer_types"][14] = "sliding_attention"
        with self.assertRaisesRegex(Gemma4PlanError, "schedule"):
            from_hf_config(cfg)
        cfg = config("12B")
        cfg["num_global_key_value_heads"] = 8
        with self.assertRaisesRegex(Gemma4PlanError, "num_global_key_value_heads"):
            from_hf_config(cfg)

    def test_projection_inventory_accounts_for_sharing_ple_and_k_eq_v(self):
        e2b = from_hf_config(config("E2B"))
        e4b = from_hf_config(config("E4B"))
        b12 = from_hf_config(config("12B"))
        self.assertEqual(len(projections(e2b)), 1 + 35 * 9 - 20 * 2)
        self.assertEqual(len(projections(e4b)), 1 + 42 * 9 - 18 * 2)
        self.assertEqual(len(projections(b12)), 48 * 7 - 8)
        self.assertIn((8960, 1536), projection_geometries(e2b))
        self.assertIn((10752, 2560), projection_geometries(e4b))
        self.assertIn((12288, 1536), projection_geometries(e2b))
        self.assertIn((1536, 12288), projection_geometries(e2b))
        self.assertIn((15360, 3840), projection_geometries(b12))
        self.assertNotIn((512, 3840), {(p.rows, p.cols) for p in projections(b12)
                                           if p.layer == 5 and p.name == "v"})
        self.assertEqual(projection_image_geometry(Projection(15, "down", 1536, 12288)),
                         (1536, 6144, 2))
        self.assertEqual(projection_image_geometry(Projection(1, "down", 2560, 10240)),
                         (2560, 5120, 2))
        self.assertEqual(projection_image_geometry(Projection(1, "down", 3840, 15360)),
                         (3840, 7680, 2))

    def test_image_plan_deduplicates_layers_and_names_split_calls(self):
        plan = image_plan(config("12B"))
        self.assertEqual(len(plan["images"]), 8)
        self.assertEqual(plan["projection_count"], 328)
        self.assertEqual(plan["images"]["n3840_k7680"]["cores"], 8)
        down = [p for p in plan["projections"] if p["layer"] == 5 and p["name"] == "down"]
        self.assertEqual((down[0]["image"], down[0]["calls"]), ("n3840_k7680", 2))


if __name__ == "__main__":
    unittest.main()
