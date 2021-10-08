### Download data

download_utils.py



### Data pre-proccessing

mind.py

* download and read data with functions:`download_mind`,`extract_mind`,`read_clickhistory`

* generate data files:`train_mind.txt`,`valid_mind_txt`,`user_history.txt`,

  with functions :`get_train_input`,`get_valid_input`,`get_user_history`

  * train file format:  `[1(pos)/0(neg)] ["train_"][userid] [pos/neg]`

    ```
    example:
    	1 train_U82271 N15368
    	0 train_U82271 N10537
    	0 train_U82271 N20663
    ```

  * valid file format: ` [1(pos)/0(neg)] ["valid_"][userid] [pos/neg]%[sess_id]`

    ```
    example:
    	1 valid_U41827 N8620%0
    	0 valid_U41827 N23699%0
    	0 valid_U41827 N21291%0
    ```

  *  user history file format: `["train_"]/["valid_"][userid] ,[clicks]`

    ```
    example:
    	train_U82271 N3130,N11621,N12917,N4574,N12140,N9748
    	train_U84185 N27209,N11723,N4617,N12320,N11333,N24461,N22111,N14026,N21705,N17551,N17039
    	train_U11552 N2139
    ```

* load words and entities with function`get_words_and_entities`

  ```
  example:
  
      news_word:
  
        The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By Shop the notebooks, jackets, and more that the royals can't live without.
  
      news_entities:
  
        [("SurfaceForms": ["Prince Charles"],"WikidataId": "Q43274"),("SurfaceForms": ["Queen Elizabeth"],"WikidataId": "Q9682")]
  ```



### Embedding

mind.py

* generate embeddings with function `get_words_and_entities`:

  generate `word_embeddings` and `entity_embeddings` with `Glove embedding`

```python
news_word_string_dict[doc_id][i] = word_index
news_entity_string_dict[doc_id][i] = entity_index
```

* generate `doc_feature.txt`:

  `[doc_id] [news_word_string_dict+","] [new_entiry_string_dict+","]`

  ```
  example:
  N3112 1,2,3,4,5,6,7,5,8,9 0,0,3,3,2,2,0,2,1,0
  N10399 1,10,11,12,13,14,15,16,1,17 0,0,0,0,0,0,0,0,0,0
  N12103 18,19,20,21,22,23,13,24,25,26 0,0,0,0,0,0,0,0,0,0
  ```

  

### Prepare hparameters

prepare_hparams.py



### Model DKN

