python3 tools/infer_kie_token_ser.py \
  -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_custom.yml \
  -o Architecture.Backbone.checkpoints=./output/ser_vi_layoutxlm_0303/best_model \
  Global.infer_img=/home/yaiba/project/KIE/dataset/data_crawl/jianshe_2/HRK202311161250460029614.jpg \
  Global.save_res_path=output/ser/jianshe_1



# python3 ./tools/infer_kie_token_ser_re.py \
#   -c configs/kie/vi_layoutxlm/re_vi_layoutxlm_xfund_zh.yml \
#   -o Architecture.Backbone.checkpoints=./pretrained_model/re_vi_layoutxlm_xfund_pretrained/best_accuracy \
#   Global.infer_img=/home/yaiba/Downloads/8434314_ios_1705892424440_f76M7x.jpg \
#   -c_ser configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml \
#   -o_ser Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy