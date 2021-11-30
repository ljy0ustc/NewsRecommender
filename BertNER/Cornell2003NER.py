import kashgari
from kashgari.corpus import Cornll2003NerCorpus
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

kashgari.config.use_cudnn_cell = True

train_x,train_y=Cornll2003NerCorpus.load_data('train')
valid_x,valid_y=Cornll2003NerCorpus.load_data('valid')
test_x,test_y=Cornll2003NerCorpus.load_data('test')

bert_embed = BERTEmbedding('./bert',task=kashgari.LABELING,sequence_length=100)


model=BiLSTM_CRF_Model(bert_embed)
model.fit(train_x,train_y,x_validate=valid_x,y_validate=valid_y,epochs=20,batch_size=512)

model.evaluate(test_x,test_y)

model.save('saved_ner_model')

loaded_model=kashgari.utils.load_model('saved_ner_model')
loaded_model.predict(test_x[:10])

loaded_model.compile_model()
model.fit(train_x,train_y,valid_x,valid_y)