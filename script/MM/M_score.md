# M_score

python3 -c "import timm; timm.create_model('coat_lite_medium.in1k', pretrained=True)"

## Backbone Model

- cait_s24_224.fb_dist_in1k
- vit_base_patch16_224.dino
- vit_large_patch14_clip_224.openai_ft_in1k
- aimv2_large_patch14_224.apple_pt_dist
- deit3_base_patch16_224.fb_in22k_ft_in1k
- beitv2_base_patch16_224.in1k_ft_in22k_in1k
- beit3_large_patch16_224.indomain_in22k_ft_in1k
- xcit_small_24_p8_224.fb_dist_in1k
- vit_base_patch16_siglip_224.v2_webli
- xcit_large_24_p8_224.fb_dist_in1k
- beitv2_large_patch16_224.in1k_ft_in22k
- vit_tiny_patch16_224.augreg_in21k_ft_in1k
- tnt_s_patch16_224.in1k
- tnt_b_patch16_224.in1k
- deit3_small_patch16_224.fb_in22k_ft_in1k

## Summary of Test Accuracies(Old)

| Code | Model Name | Test Acc |
|------|------------|----------|
| M001 | cait_s24_224.fb_dist_in1k | 0.7346 |
| M002 | vit_base_patch16_224.dino | 0.7160 |
| M003 | deit3_base_patch16_224.fb_in22k_ft_in1k | 0.7275 |
| M004 | beitv2_base_patch16_224.in1k_ft_in22k_in1k | 0.7704 |
| M005 | vit_large_patch14_clip_224.openai_ft_in1k | 0.7761 |
| M006 | aimv2_large_patch14_224.apple_pt_dist | 0.7947 |
| M007 | xcit_small_24_p8_224.fb_dist_in1k | 0.7382 |
| M008 | vit_base_patch16_siglip_224.v2_webli | 0.6638 |
| M009 | beit3_large_patch16_224.indomain_in22k_ft_in1k | 0.7847 |
| M010 | xcit_large_24_p8_224.fb_dist_in1k | 0.7332 |
| M011 | beitv2_large_patch16_224.in1k_ft_in22k | 0.7983 |
| M012 | vit_tiny_patch16_224.augreg_in21k_ft_in1k | 0.6152 |
| M015 | tnt_s_patch16_224.in1k | 0.6445 |
| M016 | tnt_b_patch16_224.in1k | 0.6896 |

## BDRFuse Results

| Code | Model Name | Test Acc |
|------|------------|----------|
| M001 | cait_s24_224.fb_dist_in1k | 0.6981 |
| M002 | vit_base_patch16_224.dino | 0.4392 |
| M003 | deit3_base_patch16_224.fb_in22k_ft_in1k | 0.7253 |
| M004 | beitv2_base_patch16_224.in1k_ft_in22k_in1k | 0.6252 |
| M005 | vit_large_patch14_clip_224.openai_ft_in1k | 0.7303 |
| M006 | aimv2_large_patch14_224.apple_pt_dist | ? |
| M007 | xcit_small_24_p8_224.fb_dist_in1k | 0.5057 |
| M008 | vit_base_patch16_siglip_224.v2_webli | 0.5408 |
| M009 | beit3_large_patch16_224.indomain_in22k_ft_in1k | 0.6624 |
| M010 | xcit_large_24_p8_224.fb_dist_in1k | ? |
| M011 | beitv2_large_patch16_224.in1k_ft_in22k | 0.6559 |
| M012 | vit_tiny_patch16_224.augreg_in21k_ft_in1k | 0.5908 |
| M015 | tnt_s_patch16_224.in1k | 0.3441 |
| M016 | tnt_b_patch16_224.in1k | 0.3584 |

## Frieren Fusion Results

| Code | Model Name | Test Acc |
|------|------------|----------|
| M001 | cait_s24_224.fb_dist_in1k | ? |
| M002 | vit_base_patch16_224.dino | ? |
| M003 | deit3_base_patch16_224.fb_in22k_ft_in1k | ? |
| M004 | beitv2_base_patch16_224.in1k_ft_in22k_in1k | ? |
| M005 | vit_large_patch14_clip_224.openai_ft_in1k | ? |
| M006 | aimv2_large_patch14_224.apple_pt_dist | ? |
| M007 | xcit_small_24_p8_224.fb_dist_in1k | ? |
| M008 | vit_base_patch16_siglip_224.v2_webli | ? |
| M009 | beit3_large_patch16_224.indomain_in22k_ft_in1k | ? |
| M010 | xcit_large_24_p8_224.fb_dist_in1k | ? |
| M011 | beitv2_large_patch16_224.in1k_ft_in22k | ? |
| M012 | vit_tiny_patch16_224.augreg_in21k_ft_in1k | ? |
| M015 | tnt_s_patch16_224.in1k | ? |
| M016 | tnt_b_patch16_224.in1k | ? |