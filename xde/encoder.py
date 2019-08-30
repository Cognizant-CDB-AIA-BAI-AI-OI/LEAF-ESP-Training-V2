class EnumStrictEncoder(object):

    @staticmethod
    def encode(target, possible_values, verbose=False):
        """
        Returns a one hot vector where the target bit is turned on if an *exact match* is found in
        the attribute list, ignoring the letter case.
        The returned one-hot vector has 1 more element than
        the passed list. This last element represents the 'other' case, and is
        turned on if the target was not found in the list
        :param target: the target value that should be turned on in the one-hot
        vector
        :param possible_values: a list of comma separated strings
        :param verbose: True if an error should be logged when the target is not found in the list of possible
        values
        :return: a one-hot vector of integers
        """
        elem_values = [v.strip().lower() for v in possible_values]
        target = target.strip().lower()
        try:
            index = elem_values.index(target)
        except ValueError as ve:
            if verbose:
                print("Unknown target value '{}'!".format(target))
                raise ve
        bit_vector = [0] * (len(elem_values))
        bit_vector[index] = 1
        return bit_vector

    @staticmethod
    def decode(encoded, possible_values):
        """
        Returns the string representation of the encoded one-hot vector
        :param encoded: a one-hot vector
        :param possible_values: a list of comma separated strings
        :return:
        """
        return possible_values[encoded.index(1.)]
