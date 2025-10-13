# ABL_score

python3 -c "import timm; timm.create_model('eva_giant_patch14_224.clip_ft_in1k', pretrained=True)"

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

## Summary of Test Accuracies

| Code | Model Name | Test Acc |
|------|------------|----------|
| ABL001 | cait_s24_224.fb_dist_in1k | 0.7096 |
| ABL002 | vit_base_patch16_224.dino | 0.6767 |
| ABL003 | deit3_base_patch16_224.fb_in22k_ft_in1k | 0.7225 |
| ABL004 | beitv2_base_patch16_224.in1k_ft_in22k_in1k | 0.7489 |
| ABL005 | vit_large_patch14_clip_224.openai_ft_in1k | ? |
| ABL006 | aimv2_large_patch14_224.apple_pt_dist | ? |
| ABL007 | xcit_small_24_p8_224.fb_dist_in1k | ? |
| M008 | vit_base_patch16_siglip_224.v2_webli | ? |
| M009 | beit3_large_patch16_224.indomain_in22k_ft_in1k | ? |
| M010 | xcit_large_24_p8_224.fb_dist_in1k | ? |
| M011 | beitv2_large_patch16_224.in1k_ft_in22k | ? |
| M012 | vit_tiny_patch16_224.augreg_in21k_ft_in1k | ? |
| M015 | tnt_s_patch16_224.in1k | ? |
| M016 | tnt_b_patch16_224.in1k | ? |