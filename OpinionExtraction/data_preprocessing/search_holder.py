import search_data
import json



if __name__=="__main__":
    file4 = 'non_fbis/08.36.15-7509'  # involvement in Kyoto has been in doubt
    file2 = '20011112/20.33.31-29984'
    file3 = 'non_fbis/04.28.09-24241'
    file1 = '20020217/20.53.10-28693' #bush, use to check the nested-host
    file5 = '20010706/02.01.27-21386' # our agency seriously needs equipment for detecting..
    file6 = '20020206/20.31.05-16359' #These words are fine words, but they do not..
    file7 = 'non_fbis/06.44.52-19992' #the brown pelican
    file8 = '20020507/22.11.06-28210'
    file9 ='20011221/20.54.40-10484'
    file10= 'ula/ch5'
    doc4=search_data.Document.from_file(file5)
    import sys
    sys.stdout = open('statistics/example.txt','a')
    print(doc4.sentence_data)
    print(doc4.ann_dict)
    print(doc4.get_nonoverlap_incomplete())
