#!/usr/bin/env
#coding:utf8
import csv
import logging

class companyScore:
    def __init__(self):
        self.company_dict = self.load_company_dictionary("./data/company.csv")
        self.company_connection_dict = self.load_company_connection_dictionary("./data/son_top_parent_dict.csv")

    def load_company_dictionary(self, path):
        company_dictionary = {}
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 0:
                        if int(row[0]) in company_dictionary and float(row[1]) > company_dictionary[int(row[0])]:
                            company_dictionary[int(row[0])] = float(row[1])
                        elif int(row[0]) not in company_dictionary:
                            company_dictionary[int(row[0])] = float(row[1])
                        else:
                            continue
                    else:
                        continue
            f.close()
        except Exception as e:
            logging.error("load_company_dictionary failed, " + e.__str__())
        return company_dictionary


    def load_company_connection_dictionary(self, path):
        company_connection_dictionary = {}
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 0:
                        company_connection_dictionary[int(row[0])] = int(row[1])
                    else:
                        continue
            f.close()
        except Exception as e:
            logging.error("load_company_connection_dictionary failed, " + e.__str__())
        return company_connection_dictionary


    def get_company_score(self, company_ids_list):
        company_score_list = []
        try:
            if len(company_ids_list) == 0:
                return company_score_list
            for company_id in set(company_ids_list):
                if company_id in self.company_connection_dict:
                    if self.company_connection_dict[company_id] in self.company_dict:
                        company_score_list.append(self.company_dict[self.company_connection_dict[company_id]])
                    else:
                        continue
                elif  company_id in self.company_dict:
                    company_score_list.append(self.company_dict[company_id])
                else:
                    company_score_list.append(0)
        except Exception as e:
            logging.error("get_company_score failed, " + e.__str__())
        return company_score_list


if __name__ == '__main__':
    '''
    logging.basicConfig(filename='./data/1.log', level=logging.WARNING,
                        format='levelname: %(levelname)s filename: %(filename)s funcName: %(funcName)s '
                               'outputNumber: [%(lineno)d]  thread: %(threadName)s output-msg: %(message)s'
                               ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]')
    '''
    company_id_list = [1250807, 1045278]
    try:
        cs = companyScore()
        score_list = cs.get_company_score(company_id_list)
        print(score_list)
    except Exception as e:
        logging.error(e.__str__())
