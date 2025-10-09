# M_score

python3 -c "import timm; timm.create_model('aimv2_1b_patch14_224.apple_pt', pretrained=True)"

## Backbone Model

- cait_s24_224.fb_dist_in1k
- vit_base_patch16_224.dino
- vit_large_patch14_clip_224.openai_ft_in1k
- aimv2_large_patch14_224.apple_pt_dist
- deit3_base_patch16_224.fb_in22k_ft_in1k
- beitv2_base_patch16_224.in1k_ft_in22k_in1k
- beit3_large_patch16_224.indomain_in22k_ft_in1k
- xcit_small_24_p8_224.fb_dist_in1k
- aimv2_1b_patch14_224.apple_pt

## Summary of Test Accuracies

| Code | Model Name | Test Acc |
|------|------------|----------|
| M001 | cait_s24_224.fb_dist_in1k | 0.7346 |
| M002 | vit_base_patch16_224.dino | 0.7146 |
| M003 | deit3_base_patch16_224.fb_in22k_ft_in1k | 0.7275 |
| M004 | beitv2_base_patch16_224.in1k_ft_in22k_in1k | 0.7654 |
| M005 | vit_large_patch14_clip_224.openai_ft_in1k | ? |
| M006 | aimv2_large_patch14_224.apple_pt_dist | ? |
| M007 | xcit_small_24_p8_224.fb_dist_in1k | ? |
| M008 | aimv2_1b_patch14_224.apple_pt | ? |
| M009 | beit3_large_patch16_224.indomain_in22k_ft_in1k | ? |