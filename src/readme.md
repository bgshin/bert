## Dataset

* [download_script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
* modified: src/download_glue_data.py

## Training

### Cola

```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.8063279
INFO:tensorflow:  eval_loss = 0.71443033
INFO:tensorflow:  global_step = 801
INFO:tensorflow:  loss = 0.720482
```

### MRPC
```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.84068626
INFO:tensorflow:  eval_loss = 0.488581
INFO:tensorflow:  global_step = 343
INFO:tensorflow:  loss = 0.488581
```

### SST-2
```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.9151376
INFO:tensorflow:  eval_loss = 0.3450427
INFO:tensorflow:  global_step = 6313
INFO:tensorflow:  loss = 0.3450427
```


# SST


## Transform dataset to tfrecord

```
python prep_sst.py # (O) -> will create tfrecord and cpkl (tokens, string, label)
python transform_to_tfrecord.py # (X)
```

* This will create

```
data/glue_data/[dataset]/[trn,dev,tst].[cased_L-12_H-768_A-12, uncased_L-24_H-1024_A-16].tf_record
```

* TODO: tf_record to txt



## Finetuning

```
python finetune_glue.py
```



##  Evaluate

```
python evaluate_glue.py
```


```
cased_L-12_H-768_A-12
INFO:tensorflow:  eval_accuracy = 0.9151376
INFO:tensorflow:  eval_loss = 0.3166457
INFO:tensorflow:  global_step = 6312
INFO:tensorflow:  loss = 0.3166457
accuracy=0.942889


uncased_L-24_H-1024_A-16
INFO:tensorflow:  eval_accuracy = 0.9266055
INFO:tensorflow:  eval_loss = 0.323713
INFO:tensorflow:  global_step = 13000
INFO:tensorflow:  loss = 0.323713
accuracy=0.919824


INFO:tensorflow:  eval_accuracy = 0.93233943
INFO:tensorflow:  eval_loss = 0.36799455
INFO:tensorflow:  global_step = 33672
INFO:tensorflow:  loss = 0.36799455
accuracy=0.940692
```


##  Generate teaching materials

```
python generate_materials.py
```

* embedding, x, y, ysm(y softmax)


## distill with cnn

```
python distill_cnn.py
```




# Time

```
[cased base]
======================= trn =========================
67349it [06:28, 173.30it/s]
======================= dev =========================
872it [00:09, 93.97it/s]
======================= tst =========================
1821it [00:14, 128.05it/s]
0.942888522789676

67349+872+1821 = 70042
6:28+9+14 = 6:51 = 411s
70042/411 = 170.418it/s

[uncased large]
======================= trn =========================
67349it [17:49, 62.95it/s]
======================= dev =========================
872it [00:23, 37.39it/s]
======================= tst =========================
1821it [00:36, 50.41it/s]
0.9406919275123559

17:49+23+36 = 108+17*60 =1128
70042/1128 = 62.09it/s
```



# Keras Bert Project

```
[Embedding]
bert/embeddings/LayerNorm/beta -> Embedding-Norm
bert/embeddings/LayerNorm/gamma -> Embedding-Norm
bert/embeddings/position_embeddings -> Embedding-Position
bert/embeddings/token_type_embeddings -> Embedding-Segment
bert/embeddings/word_embeddings -> Embedding-Token

[Encoder-%d-MultiHeadSelfAttention]
bert/encoder/layer_0/attention/self/query/kernel
bert/encoder/layer_0/attention/self/query/bias
bert/encoder/layer_0/attention/self/key/kernel
bert/encoder/layer_0/attention/self/key/bias
bert/encoder/layer_0/attention/self/value/kernel
bert/encoder/layer_0/attention/self/value/bias
bert/encoder/layer_0/attention/output/dense/kernel
bert/encoder/layer_0/attention/output/dense/bias


[Encoder-%d-MultiHeadSelfAttention-Norm]
bert/encoder/layer_0/attention/output/LayerNorm/gamma
bert/encoder/layer_0/attention/output/LayerNorm/beta


[Encoder-%d-FeedForward]
bert/encoder/layer_0/intermediate/dense/kernel
bert/encoder/layer_0/intermediate/dense/bias
bert/encoder/layer_0/output/dense/kernel
bert/encoder/layer_0/output/dense/bias

[Encoder-%d-FeedForward-Norm]
bert/encoder/layer_0/output/LayerNorm/gamma
bert/encoder/layer_0/output/LayerNorm/beta

...



[MLM-Dense]
cls/predictions/transform/dense/kernel
cls/predictions/transform/dense/bias

[MLM-Norm]
cls/predictions/transform/LayerNorm/gamma
cls/predictions/transform/LayerNorm/beta

[MLM-Sim]
cls/predictions/output_bias

[NSP-Dense]
bert/pooler/dense/kernel
bert/pooler/dense/bias
# We "pool" the model by simply taking the hidden state corresponding
# to the first token. We assume that this has been pre-trained
# result -> self.pooled_output

[NSP]
cls/seq_relationship/output_weights
cls/seq_relationship/output_bias


```

## classifier

* dropout(0.9)
* crossentrophy(WX+b, y)




* extimator.get_variable_names()

```
bert/loss/output_weights
bert/loss/output_bias
```





### tf-version assignment_map

```

bert/embeddings/LayerNorm/beta: bert/embeddings/LayerNorm/beta
bert/embeddings/LayerNorm/gamma: bert/embeddings/LayerNorm/gamma
bert/embeddings/position_embeddings: bert/embeddings/position_embeddings
bert/embeddings/token_type_embeddings: bert/embeddings/token_type_embeddings
bert/embeddings/word_embeddings: bert/embeddings/word_embeddings
bert/encoder/layer_0/attention/output/LayerNorm/beta: bert/encoder/layer_0/attention/output/LayerNorm/beta
bert/encoder/layer_0/attention/output/LayerNorm/gamma: bert/encoder/layer_0/attention/output/LayerNorm/gamma
bert/encoder/layer_0/attention/output/dense/bias: bert/encoder/layer_0/attention/output/dense/bias
bert/encoder/layer_0/attention/output/dense/kernel: bert/encoder/layer_0/attention/output/dense/kernel
bert/encoder/layer_0/attention/self/key/bias: bert/encoder/layer_0/attention/self/key/bias
bert/encoder/layer_0/attention/self/key/kernel: bert/encoder/layer_0/attention/self/key/kernel
bert/encoder/layer_0/attention/self/query/bias: bert/encoder/layer_0/attention/self/query/bias
bert/encoder/layer_0/attention/self/query/kernel: bert/encoder/layer_0/attention/self/query/kernel
bert/encoder/layer_0/attention/self/value/bias: bert/encoder/layer_0/attention/self/value/bias
bert/encoder/layer_0/attention/self/value/kernel: bert/encoder/layer_0/attention/self/value/kernel
bert/encoder/layer_0/intermediate/dense/bias: bert/encoder/layer_0/intermediate/dense/bias
bert/encoder/layer_0/intermediate/dense/kernel: bert/encoder/layer_0/intermediate/dense/kernel
bert/encoder/layer_0/output/LayerNorm/beta: bert/encoder/layer_0/output/LayerNorm/beta
bert/encoder/layer_0/output/LayerNorm/gamma: bert/encoder/layer_0/output/LayerNorm/gamma
bert/encoder/layer_0/output/dense/bias: bert/encoder/layer_0/output/dense/bias
bert/encoder/layer_0/output/dense/kernel: bert/encoder/layer_0/output/dense/kernel
...
bert/pooler/dense/bias: bert/pooler/dense/bias
bert/pooler/dense/kernel: bert/pooler/dense/kernel
```
