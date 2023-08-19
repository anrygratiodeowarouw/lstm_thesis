import pandas
import numpy as np

class DataProcessor:
    def __init__(self, dataframe, diameter, COL, l):
        # ... same initialization code ...

    def get_bobot(self, data):
        # ... same function code ...

    def weighted_average(self, data, bobot):
        # ... same function code ...

    def get_all_value(self):
        self.ns1 = self.weighted_average(self.bagian_1['n-spt'], self.bagian_1['bobot'])
        self.s1 = self.bagian_1['group_soil'].value_counts().index[0]
        self.ns2 = self.weighted_average(self.bagian_2['n-spt'], self.bagian_2['bobot'])
        self.s2 = self.bagian_2['group_soil'].value_counts().index[0]
        self.ns3 = self.weighted_average(self.bagian_3['n-spt'], self.bagian_3['bobot'])
        self.s3 = self.bagian_3['group_soil'].value_counts().index[0]
        self.ns4 = self.weighted_average(self.bagian_4['n-spt'], self.bagian_4['bobot'])
        self.s4 = self.bagian_4['group_soil'].value_counts().index[0]
        self.ns5 = self.weighted_average(self.bagian_5['n-spt'], self.bagian_5['bobot'])
        self.s5 = self.bagian_5['group_soil'].value_counts().index[0]
        self.nt = self.weighted_average(self.data_nt['n-spt'], self.data_nt['bobot'])
        self.st = self.data_nt['group_soil'].value_counts().index[0]
        return {
            'ns1': self.ns1,
            's1': self.s1,
            'ns2': self.ns2,
            's2': self.s2,
            'ns3': self.ns3,
            's3': self.s3,
            'ns4': self.ns4,
            's4': self.s4,
            'ns5': self.ns5,
            's5': self.s5,
            'nt': self.nt,
            'st': self.st
        }
    
    def get_data_used(self):
        return self.data_used
