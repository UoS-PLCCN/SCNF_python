from scnfn.scnf.eval import eval_disjunction


class TestEvalDisjunction():
    LITERALS = ["gene1", "gene2", "gene3"]

    def test_true_literal(self):
        state = [False, False, False]
        disjunction = ["gene1", "True", "gene3"]
        assert eval_disjunction(state, disjunction, self.LITERALS) is True

    def test_negation(self):
        state = [False, False, False]
        disjunction = ["gene1", "~gene2", "gene2"]
        assert eval_disjunction(state, disjunction, self.LITERALS) is True

    def test_false(self):
        state = [False, False, True]
        disjunction = ["gene1", "gene2", "~gene3"]
        assert eval_disjunction(state, disjunction, self.LITERALS) is False
