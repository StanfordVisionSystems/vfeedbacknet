#!/usr/bin/python3

import json
import numpy as np

taxonomy = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 3,
    11: 3,
    12: 4,
    13: 4,
    14: 4,
    15: 4,
    16: 5,
    17: 5,
    18: 5,
    19: 5,
    20: 6,
    21: 6,
    22: 7,
    23: 8,
    24: 9,
    25: 10,
    26: 11
}

# label = {
#     0:  'Swiping Left',
#     1:  'Swiping Right',
#     2:  'Swiping Down',
#     3:  'Swiping Up',
#     4:  'Pushing Hand Away',
#     5:  'Pulling Hand In',
#     6:  'Sliding Two Fingers Left',
#     7:  'Sliding Two Fingers Right',
#     8:  'Sliding Two Fingers Down',
#     9:  'Sliding Two Fingers Up',
#     10: 'Pushing Two Fingers Away',
#     11: 'Pulling Two Fingers In',
#     12: 'Rolling Hand Forward',
#     13: 'Rolling Hand Backward',
#     14: 'Turning Hand Clockwise',
#     15: 'Turning Hand Counterclockwise',
#     16: 'Zooming In With Full Hand',
#     17: 'Zooming Out With Full Hand',
#     18: 'Zooming In With Two Fingers',
#     19: 'Zooming Out With Two Fingers',
#     20: 'Thumb Up',
#     21: 'Thumb Down',
#     22: 'Shaking Hand',
#     23: 'Stop Sign',
#     24: 'Drumming Fingers',
#     25: 'No gesture',
#     26: 'Doing other things'
# }
    
with open('feedbacknet_20bn_jester.log', 'r') as f:
#with open('feedbacknet_ucf101.log', 'r') as f:
    lines = list(filter(lambda x: '[' in x, f.read().strip().split('\n')))

    assert len(lines) % 3 == 0, 'number of lines must be multiple of 3'

    data = []
    for i in range(0, len(lines), 3):
        fb1 = lines[i]
        fb2 = lines[i+1]
        fb3 = lines[i+2]

        # get necessary information
        true_label = int(fb1.split('true_label,prediction: ')[1].split(',')[0])

        fb1_top5 = list(map(lambda x: int(x), eval('['+ fb1.split('([')[1].split('])')[0] +']')))
        fb2_top5 = list(map(lambda x: int(x), eval('['+ fb2.split('([')[1].split('])')[0] +']')))
        fb3_top5 = list(map(lambda x: int(x), eval('['+ fb3.split('([')[1].split('])')[0] +']')))

        fb1_frames_top1 = list(map(lambda x: int(x), eval('['+ fb1.split(']) [')[1].split(']')[0] +']')))
        fb2_frames_top1 = list(map(lambda x: int(x), eval('['+ fb2.split(']) [')[1].split(']')[0] +']')))
        fb3_frames_top1 = list(map(lambda x: int(x), eval('['+ fb3.split(']) [')[1].split(']')[0] +']')))

        data.append({
            'true_label' : true_label,
            'num_frames' : len(fb1_frames_top1),
            'fb1_top5' : fb1_top5,
            'fb2_top5' : fb2_top5,
            'fb3_top5' : fb3_top5,
            'fb1_frames_top1' : fb1_frames_top1,
            'fb2_frames_top1' : fb2_frames_top1,
            'fb3_frames_top1' : fb3_frames_top1,
        })

    # perform analysis
    # How does top1 prediction accuracy change with feedback iterations?
    # Do the predictions follow a taxonomy?
    # Do more feedback iterations make it possible to watch less frames?
    # no temporal information...

    correct_bins1 = [ 0 for _ in range(20) ] # a bin for every 5% of video
    correct_bins2 = [ 0 for _ in range(20) ] # a bin for every 5% of video
    correct_bins3 = [ 0 for _ in range(20) ] # a bin for every 5% of video
    for d in data:
        true_label = d['true_label'] 

        def alloc(lower, upper, l):

            for j in range(int(np.floor(lower)), int(np.ceil(upper))):
                bot = max(j, lower)
                top = min(j+1, upper)
                l[j] += top-bot
                
        length = d['num_frames']
        for i in range(length):
            lower = 20*(i/length)
            upper = 20*((i+1)/length)

            if d['fb1_frames_top1'][i] == true_label:
                alloc(lower, upper, correct_bins1)
            if d['fb2_frames_top1'][i] == true_label:
                alloc(lower, upper, correct_bins2)
            if d['fb3_frames_top1'][i] == true_label:
                alloc(lower, upper, correct_bins3)

    for i in range(20):
        print(correct_bins1[i] / len(data),  end='\t')
        print(correct_bins2[i] / len(data),  end='\t')
        print(correct_bins3[i] / len(data),  end='\n') 
    print()
    for i in range(20):
        print(correct_bins2[i] / len(data), ',',  end='')
    print()
    for i in range(20):
        print(correct_bins3[i] / len(data), ',',  end='') 
    print()
               
    results = {}

    def topN(key):
        
        top1_count = 0 
        top3_count = 0 
        top5_count = 0 
        for d in data:
            if d['true_label'] == d['{}_top5'.format(key)][0]:
                top1_count += 1
                top3_count += 1
                top5_count += 1
                continue

            elif d['true_label'] in d['{}_top5'.format(key)][0:3]:
                top3_count += 1
                top5_count += 1
                continue

            elif d['true_label'] in d['{}_top5'.format(key)]:
                top5_count += 1
                continue
        r = {}
        r['top1'] = top1_count/len(data)
        r['top3'] = top3_count/len(data)
        r['top5'] = top5_count/len(data)

        return r
        
    t1 = topN('fb1')
    t2 = topN('fb2')
    t3 = topN('fb3')

    results['top1'] = {
        'fb1' : t1['top1'],
        'fb2' : t2['top1'],
        'fb3' : t3['top1'],
        'comment' : 'the top-1 accuracy after watching the whole video'
    }
    results['top3'] = {
        'fb1' : t1['top5'],
        'fb2' : t2['top5'],
        'fb3' : t3['top5'],
        'comment' : 'the top-3 accuracy after watching the whole video'
    }
    results['top5'] = {
        'fb1' : t1['top5'],
        'fb2' : t2['top5'],
        'fb3' : t3['top5'],
        'comment' : 'the top-5 accuracy after watching the whole video'
    }

    def no_temporal(key):
        count = 0 
        for d in data:
            if d['true_label'] == d['{}_frames_top1'.format(key)][0]:
                count += 1

        return count / len(data)
        
    nt1 = no_temporal('fb1')
    nt2 = no_temporal('fb2')
    nt3 = no_temporal('fb3')

    results['top1_no_temporal'] = {
        'fb1' : nt1,
        'fb2' : nt2,
        'fb3' : nt3,
        'comment' : 'the top-1 accuracy after looking at only the first frame'
    }
    
    def earliest_correct_fb(key):
        '''
        assuming the NN got the final answer correct by the third 
        '''
        
        percentage_of_video_watched = []
        for d in data:
            if d['true_label'] != d['{}_top5'.format(key)][0]:
                continue
            #if d['true_label'] != d['fb1_top5'.format(key)][0] and d['true_label'] != d['fb2_top5'.format(key)][0] and d['true_label'] != d['fb3_top5'.format(key)][0]:
            #    continue

            last_incorrect = 0
            for i in range(len(d['{}_frames_top1'.format(key)])):
                if d['{}_frames_top1'.format(key)][i] != d['true_label']:
                    last_incorrect = i


            percentage_of_video_watched.append((last_incorrect+1) / len(d['{}_frames_top1'.format(key)]))

        mean = np.mean(percentage_of_video_watched)
        median = np.median(percentage_of_video_watched)

        return {
            'mean' : mean,
            'median' : median
        }

    p1 = earliest_correct_fb('fb1')
    p2 = earliest_correct_fb('fb2')
    p3 = earliest_correct_fb('fb3')

    results['mean_video_watched'] = { 
        'fb1' : p1['mean'],
        'fb2' : p2['mean'],
        'fb3' : p3['mean'],
        'comment' : 'the mean percentage of the video that needed to be watched to become correct until the end of the clip'    
    }        


    results['median_video_watched'] = { 
        'fb1' : p1['median'],
        'fb2' : p2['median'],
        'fb3' : p3['median'],
        'comment' : 'the median percentage of the video that needed to be watched to become correct until the end of the clip'    
    }        

    # def taxonomic_classification(key):

    #     percentage_in_class = []
    #     for d in data:
    #         true_class = taxonomy[d['true_label']]

    #         count = 0
    #         for item in d['{}_top5'.format(key)]:
    #             if taxonomy[item] == true_class:
    #                 count += 1

    #         percentage_in_class.append(count / 5)

    #     mean = np.mean(percentage_in_class)
    #     median = np.median(percentage_in_class)

    #     return {
    #         'mean' : mean,
    #         'median' : median
    #     }

    # tr1 = taxonomic_classification('fb1')
    # tr2 = taxonomic_classification('fb2')
    # tr3 = taxonomic_classification('fb3')

    # # results['mean_taxonomy'] = {
    # #     'fb1' : tr1['mean'],
    # #     'fb2' : tr2['mean'],
    # #     'fb3' : tr3['mean']
    # # }
    # # results['median_taxonomy'] = {
    # #     'fb1' : tr1['median'],
    # #     'fb2' : tr2['median'],
    # #     'fb3' : tr3['median']
    # # }

    # def taxonomic_compliance(key):

    #     correct = 0
    #     for d in data:
    #         true_class = taxonomy[d['true_label']]

    #         if true_class == taxonomy[d['{}_top5'.format(key)][0]]:
    #             correct += 1
            
    #     return correct / len(data)
    
    # tc1 = taxonomic_compliance('fb1')
    # tc2 = taxonomic_compliance('fb2')
    # tc3 = taxonomic_compliance('fb3')

    # results['top1_taxonomy_compliance'] = {
    #     'fb1' : tc1,
    #     'fb2' : tc2,
    #     'fb3' : tc3,
    #     'comment' : 'using high order classes to compute accuracy, what is the top-1 compliance/accuracy of the NN after watching the whole video'
    # }
    
    print(json.dumps(results, indent=4, sort_keys=True))
