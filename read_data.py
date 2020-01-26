def _read_data(input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            if len(contends) == 0:
                assert len(words) == len(labels)
                if len(words) > 30:
                    while len(words) > 30:
                        tmplabel = labels[:30]
                        for iidx in range(len(tmplabel)):
                            if tmplabel.pop() == 'O':
                                 break
                        l = ' '.join([label for label in labels[:len(tmplabel)+1] if len(label) > 0])
                        w = ' '.join([word for word in words[:len(tmplabel)+1] if len(word) > 0])
                        lines.append([l, w])
                        words = words[len(tmplabel)+1:]
                        labels = labels[len(tmplabel)+1:]

                if len(words) == 0:
                    continue

                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])

                lines.append([l, w])
                #print(lines)
                words = []
                labels = []
                continue
            word = line.strip().split()[0]
            label = line.strip().split()[-1]
            words.append(word)
            labels.append(label)
        if len(words) > 0:
            while len(words) > 30:
                tmplabel = labels[:30]
                for iidx in range(len(tmplabel)):
                    if tmplabel.pop() == 'O':
                         break
                l = ' '.join([label for label in labels[:len(tmplabel)+1] if len(label) > 0])
                w = ' '.join([word for word in words[:len(tmplabel)+1] if len(word) > 0])
                lines.append([l, w])
                words = words[len(tmplabel)+1:]
                labels = labels[len(tmplabel)+1:]
            if len(words) == 0:
                return lines

            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append([l, w])
        return lines
lines = _read_data("toy_ner_data.tsv")
print(lines)
