## single
raygun-sample-single --minlength 180 --maxlength 200 --finetune --finetune-trainf train.fasta --finetune-validf test.fasta --finetune-epochs 1 template.fasta samples-180-200-with-ftuning

## multiple
raygun-sample-multiple --leninfo leninfo.json --finetune --finetune-trainf train.fasta --finetune-validf test.fasta --finetune-epochs 1 template.fasta samples-180-200-with-ftuning