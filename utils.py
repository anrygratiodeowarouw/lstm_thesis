import pandas
import numpy as np

class DataProcessor:
    def __init__(self, dataframe, diameter, COL, l):
        self.data = dataframe
        self.diameter = diameter
        self.COL = COL
        self.l = l
        self.grouping = {'sand': ['SP', 'SW', 'SC', 'SM', "GW", 'GP'],
                         'clay': ['CH', 'CL'],
                         'silt': ['MH', 'ML']}
        self.data['group_soil'] = self.data['soil_type'].apply(lambda x: [key for key, value in self.grouping.items() if x in value][0])
        self.data['bobot'] = self.get_bobot(self.data['depth'])
        self.data_used = self.data[(self.data['depth'] < self.l + self.COL) & (self.data['depth'] > self.COL)]
        self.bagian = (self.data_used['depth'].max() - self.data_used['depth'].min()) / 5
        self.bagian_1 = self.data_used[self.data_used['depth'] < self.data_used['depth'].min() + self.bagian]
        self.bagian_2 = self.data_used[(self.data_used['depth'] < self.data_used['depth'].min() + self.bagian * 2) & (self.data_used['depth'] > self.data_used['depth'].min() + self.bagian)]
        self.bagian_3 = self.data_used[(self.data_used['depth'] < self.data_used['depth'].min() + self.bagian * 3) & (self.data_used['depth'] > self.data_used['depth'].min() + self.bagian * 2)]
        self.bagian_4 = self.data_used[(self.data_used['depth'] < self.data_used['depth'].min() + self.bagian * 4) & (self.data_used['depth'] > self.data_used['depth'].min() + self.bagian * 3)]
        self.bagian_5 = self.data_used[self.data_used['depth'] > self.data_used['depth'].min() + self.bagian * 4]
        self.batas_atas = 4 * self.diameter if self.data_used['n-spt'].iloc[-1] >= 20 else 8 * self.diameter
        self.batas_bawah = 2 * self.diameter if self.data_used['n-spt'].iloc[-1] >= 20 else 4 * self.diameter
        self.length_nt_1 = self.l - self.batas_atas
        self.length_nt_2 = self.l + self.batas_bawah
        self.data_nt = self.data[(self.data['depth'] < self.length_nt_2) & (self.data['depth'] > self.length_nt_1)]

    def get_bobot(self, data):
        out = []
        for i in range(len(data)):
            if i == 0:
                out.append(1)
            else:
                out.append(data[i]-data[i-1])

        return np.array(out)
    
    def weighted_average(self, data, bobot):
        return np.sum(data*bobot)/np.sum(bobot)
    
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
