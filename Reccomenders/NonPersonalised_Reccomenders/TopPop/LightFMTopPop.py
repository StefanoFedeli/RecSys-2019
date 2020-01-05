def loadOutputs(fp):
    fp.seek(19)
    outputs = {}
    for line in fp:
        split = line.split(",")
        outputs[(int(split[0]))] = list(map(int, split[1].split()))
    return outputs

class LightFMTopPopRecommender(object):

    def fit(self):
        with open('../../../Outputs/LightFM_topPop_3_1200_all.csv', 'r') as fp:
            self.outputs = loadOutputs(fp)

    def recommend(self, user_id, at=10, remove_seen=True):
        return self.outputs[user_id]