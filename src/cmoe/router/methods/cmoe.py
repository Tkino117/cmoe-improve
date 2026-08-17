"""現行 CMoE のルーター。すべての実験の対照であり、既定でもある。"""


class CMoEMethod:
    name = 'cmoe'
    training_free = True
    requires_source_weights = False

    def build(self, context, baseline):
        return baseline
