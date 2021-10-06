## Bert+BiLSTM+CRF NER with Kashgari

#### BiLSTM+CRF labeling

```
class BiLSTM_Model(ABCLabelingModel):
    @classmethod
    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_stack = [
            L.Bidirectional(L.LSTM(**config['layer_blstm']), name='layer_blstm'),
            L.Dropout(**config['layer_dropout'], name='layer_dropout'),
            L.Dense(output_dim, **config['layer_time_distributed']),
            L.Activation(**config['layer_activation'])
        ]
        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)
```



#### Read Corpus

##### ChineseDailyNerCorpus：

```
class ChineseDailyNerCorpus:
    """
    Chinese Daily New New Corpus
    https://github.com/zjy-ucas/ChineseNER/

    Example:
        >>> from kashgari.corpus import ChineseDailyNerCorpus
        >>> train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        >>> test_x, test_y = ChineseDailyNerCorpus.load_data('test')
        >>> valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        >>> print(train_x)
            [['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', ...], ...]
        >>> print(train_y)
            [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', ...], ...]
    """
    __corpus_name__ = 'china-people-daily-ner-corpus'
    __zip_file__name = 'http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz'

    @classmethod
    def load_data(cls,
                  subset_name: str = 'train',
                  shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

        Args:
            subset_name: {train, test, valid}
            shuffle: should shuffle or not, default True.

        Returns:
            dataset_features and dataset labels
        """
        corpus_path = get_file(cls.__corpus_name__,
                               cls.__zip_file__name,
                               cache_dir=K.DATA_PATH,
                               untar=True)

        if subset_name == 'train':
            file_path = os.path.join(corpus_path, 'example.train')
        elif subset_name == 'test':
            file_path = os.path.join(corpus_path, 'example.test')
        else:
            file_path = os.path.join(corpus_path, 'example.dev')

        x_data, y_data = DataReader.read_conll_format_file(file_path)
        if shuffle:
            x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)
        logger.debug(f"loaded {len(x_data)} samples from {file_path}. Sample:\n"
                     f"x[0]: {x_data[0]}\n"
                     f"y[0]: {y_data[0]}")
        return x_data, y_data
```

##### Cornll2003NerCorpus：

```
class ChineseDailyNerCorpus:
    """
    Chinese Daily New New Corpus
    https://github.com/zjy-ucas/ChineseNER/

    Example:
        >>> from kashgari.corpus import ChineseDailyNerCorpus
        >>> train_x, train_y = ChineseDailyNerCorpus.load_data('train')
        >>> test_x, test_y = ChineseDailyNerCorpus.load_data('test')
        >>> valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
        >>> print(train_x)
            [['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', ...], ...]
        >>> print(train_y)
            [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', ...], ...]
    """
    __corpus_name__ = 'china-people-daily-ner-corpus'
    __zip_file__name = 'http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz'

    @classmethod
    def load_data(cls,
                  subset_name: str = 'train',
                  shuffle: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load dataset as sequence labeling format, char level tokenized

        Args:
            subset_name: {train, test, valid}
            shuffle: should shuffle or not, default True.

        Returns:
            dataset_features and dataset labels
        """
        corpus_path = get_file(cls.__corpus_name__,
                               cls.__zip_file__name,
                               cache_dir=K.DATA_PATH,
                               untar=True)

        if subset_name == 'train':
            file_path = os.path.join(corpus_path, 'example.train')
        elif subset_name == 'test':
            file_path = os.path.join(corpus_path, 'example.test')
        else:
            file_path = os.path.join(corpus_path, 'example.dev')

        x_data, y_data = DataReader.read_conll_format_file(file_path)
        if shuffle:
            x_data, y_data = utils.unison_shuffled_copies(x_data, y_data)
        logger.debug(f"loaded {len(x_data)} samples from {file_path}. Sample:\n"
                     f"x[0]: {x_data[0]}\n"
                     f"y[0]: {y_data[0]}")
        return x_data, y_data
```

