# N-shot optimization

```bash
(venv) aelaguiz@Amirs-MacBook-Pro langdspy % python examples/amazon/generate_slugs.py model.pkl
Hit enter to evaluate the untrained model...
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:05<00:00,  7.20it/s]
Before Training Accuracy: 0.8194524481637916
Hit enter to train the model...
2024-03-07 22:09:39,597 - DEBUG - Total number of examples: 2 Example size: 103 n_examples: 2 example_X size: 103 Scoring size: 45
Evaluating subsets:   0%|                                                                                                                                                                                                                                                   | 0/100 [00:00<?, ?it/s]2024-03-07 22:09:45,930 - DEBUG - Training subset scored 0.8785408655092349
Evaluating subsets:   8%|██████████████████▊                                                                                                                                                                                                                        | 8/100 [00:06<01:12,  1.26it/s]2024-03-07 22:09:46,185 - DEBUG - Training subset scored 0.8814046076709651
2024-03-07 22:09:46,491 - DEBUG - Training subset scored 0.8760986649928661
2024-03-07 22:09:46,895 - DEBUG - Training subset scored 0.8764235724331444
2024-03-07 22:09:51,587 - DEBUG - Training subset scored 0.8934962682978934
Evaluating subsets:  12%|████████████████████████████                                                                                                                                                                                                              | 12/100 [00:11<01:32,  1.05s/it]2024-03-07 22:09:52,284 - DEBUG - Training subset scored 0.8801273094400404
2024-03-07 22:09:52,971 - DEBUG - Training subset scored 0.8830420810976756
2024-03-07 22:09:53,087 - DEBUG - Training subset scored 0.8903506052569576
2024-03-07 22:09:57,761 - DEBUG - Training subset scored 0.8625979046589743
Evaluating subsets:  16%|█████████████████████████████████████▍                                                                                                                                                                                                    | 16/100 [00:18<01:43,  1.23s/it]2024-03-07 22:09:58,441 - DEBUG - Training subset scored 0.8829796555085418
2024-03-07 22:09:59,215 - DEBUG - Training subset scored 0.8744870418250539
2024-03-07 22:09:59,812 - DEBUG - Training subset scored 0.858287858911917
2024-03-07 22:10:04,455 - DEBUG - Training subset scored 0.8612618002507423
Evaluating subsets:  20%|██████████████████████████████████████████████▊                                                                                                                                                                                           | 20/100 [00:24<01:50,  1.39s/it]2024-03-07 22:10:05,642 - DEBUG - Training subset scored 0.8661274325222308
2024-03-07 22:10:06,664 - DEBUG - Training subset scored 0.85168947285815
2024-03-07 22:10:10,446 - DEBUG - Training subset scored 0.8711482842387782
2024-03-07 22:10:11,835 - DEBUG - Training subset scored 0.8832097977107368
Evaluating subsets:  24%|████████████████████████████████████████████████████████▏                                                                                                                                                                                 | 24/100 [00:32<01:56,  1.54s/it]2024-03-07 22:10:12,456 - DEBUG - Training subset scored 0.8778663444680921
2024-03-07 22:10:13,932 - DEBUG - Training subset scored 0.8481657104788697
2024-03-07 22:10:16,554 - DEBUG - Training subset scored 0.8643763770246858
2024-03-07 22:10:17,967 - DEBUG - Training subset scored 0.87195558132136
Evaluating subsets:  28%|█████████████████████████████████████████████████████████████████▌                                                                                                                                                                        | 28/100 [00:38<01:50,  1.54s/it]2024-03-07 22:10:19,451 - DEBUG - Training subset scored 0.8683955324592978
2024-03-07 22:10:23,846 - DEBUG - Training subset scored 0.8679790230174045
2024-03-07 22:10:25,288 - DEBUG - Training subset scored 0.8713728557146283
2024-03-07 22:10:26,655 - DEBUG - Training subset scored 0.8849326922126938
Evaluating subsets:  32%|██████████████████████████████████████████████████████████████████████████▉                                                                                                                                                               | 32/100 [00:47<01:58,  1.74s/it]2024-03-07 22:10:29,871 - DEBUG - Training subset scored 0.8734383450427299
2024-03-07 22:10:32,235 - DEBUG - Training subset scored 0.8725723629413675
2024-03-07 22:10:36,958 - DEBUG - Training subset scored 0.8729666267874104
2024-03-07 22:10:39,043 - DEBUG - Training subset scored 0.8898426184308865
Evaluating subsets:  36%|████████████████████████████████████████████████████████████████████████████████████▏                                                                                                                                                     | 36/100 [00:59<02:18,  2.16s/it]2024-03-07 22:10:41,359 - DEBUG - Training subset scored 0.87421944051681
2024-03-07 22:10:43,449 - DEBUG - Training subset scored 0.8677940341881134
2024-03-07 22:10:44,942 - DEBUG - Training subset scored 0.8800687073657738
2024-03-07 22:10:47,268 - DEBUG - Training subset scored 0.8682598626320116
Evaluating subsets:  40%|█████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                            | 40/100 [01:07<02:07,  2.13s/it]2024-03-07 22:10:49,733 - DEBUG - Training subset scored 0.8623343918931639
2024-03-07 22:10:51,682 - DEBUG - Training subset scored 0.8702139729030367
2024-03-07 22:10:53,728 - DEBUG - Training subset scored 0.8690090186541871
2024-03-07 22:10:55,461 - DEBUG - Training subset scored 0.8679486803591547
Evaluating subsets:  44%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                                                   | 44/100 [01:15<01:57,  2.10s/it]2024-03-07 22:10:57,924 - DEBUG - Training subset scored 0.8794691601938754
2024-03-07 22:10:58,151 - DEBUG - Training subset scored 0.859370236879358
2024-03-07 22:11:00,069 - DEBUG - Training subset scored 0.8729378654669147
2024-03-07 22:11:01,426 - DEBUG - Training subset scored 0.8708572953072279
Evaluating subsets:  48%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                                                         | 48/100 [01:21<01:39,  1.92s/it]2024-03-07 22:11:03,762 - DEBUG - Training subset scored 0.8829605368274817
2024-03-07 22:11:04,222 - DEBUG - Training subset scored 0.8696982105076134
2024-03-07 22:11:05,938 - DEBUG - Training subset scored 0.8920328839059286
2024-03-07 22:11:07,400 - DEBUG - Training subset scored 0.8774790135143933
Evaluating subsets:  52%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                | 52/100 [01:27<01:25,  1.79s/it]2024-03-07 22:11:09,585 - DEBUG - Training subset scored 0.8645401777631032
2024-03-07 22:11:10,074 - DEBUG - Training subset scored 0.8570089703477265
2024-03-07 22:11:11,798 - DEBUG - Training subset scored 0.8754117777277584
2024-03-07 22:11:13,663 - DEBUG - Training subset scored 0.8396668473978952
Evaluating subsets:  56%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                       | 56/100 [01:34<01:15,  1.72s/it]2024-03-07 22:11:15,970 - DEBUG - Training subset scored 0.8714956215879119
2024-03-07 22:11:16,987 - DEBUG - Training subset scored 0.8577755235693728
2024-03-07 22:11:18,186 - DEBUG - Training subset scored 0.8593159589174947
2024-03-07 22:11:19,621 - DEBUG - Training subset scored 0.8796957579101851
Evaluating subsets:  60%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                             | 60/100 [01:40<01:06,  1.65s/it]2024-03-07 22:11:23,134 - DEBUG - Training subset scored 0.8755527399440635
2024-03-07 22:11:25,236 - DEBUG - Training subset scored 0.8670879691010577
2024-03-07 22:11:29,557 - DEBUG - Training subset scored 0.86591179777457
2024-03-07 22:11:31,018 - DEBUG - Training subset scored 0.8815842850871003
Evaluating subsets:  64%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                    | 64/100 [01:51<01:12,  2.01s/it]2024-03-07 22:11:35,352 - DEBUG - Training subset scored 0.8787773128303796
2024-03-07 22:11:36,706 - DEBUG - Training subset scored 0.8843122254690964
2024-03-07 22:11:40,925 - DEBUG - Training subset scored 0.8807765365214743
2024-03-07 22:11:42,174 - DEBUG - Training subset scored 0.8633645818508353
Evaluating subsets:  68%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                           | 68/100 [02:02<01:11,  2.25s/it]2024-03-07 22:11:50,031 - DEBUG - Training subset scored 0.8749369738665906
2024-03-07 22:12:17,714 - DEBUG - Training subset scored 0.8587455588769662
2024-03-07 22:12:20,122 - DEBUG - Training subset scored 0.8497660630051325
2024-03-07 22:12:23,456 - DEBUG - Training subset scored 0.8424987322474388
Evaluating subsets:  72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                 | 72/100 [02:43<02:10,  4.67s/it]2024-03-07 22:12:26,013 - DEBUG - Training subset scored 0.8597523169680503
2024-03-07 22:12:29,180 - DEBUG - Training subset scored 0.8748033052157567
2024-03-07 22:12:32,077 - DEBUG - Training subset scored 0.8552352426207824
2024-03-07 22:12:35,773 - DEBUG - Training subset scored 0.8704678552431757
Evaluating subsets:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                        | 76/100 [02:56<01:40,  4.19s/it]2024-03-07 22:12:38,067 - DEBUG - Training subset scored 0.8800391309755009
2024-03-07 22:12:39,789 - DEBUG - Training subset scored 0.848924486990501
2024-03-07 22:12:41,353 - DEBUG - Training subset scored 0.8790530610551548
2024-03-07 22:12:43,920 - DEBUG - Training subset scored 0.8794261748248888
Evaluating subsets:  80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                              | 80/100 [03:04<01:10,  3.55s/it]2024-03-07 22:12:45,413 - DEBUG - Training subset scored 0.8889178071440995
2024-03-07 22:12:49,637 - DEBUG - Training subset scored 0.8741902479565049
2024-03-07 22:12:51,398 - DEBUG - Training subset scored 0.8781555961320465
2024-03-07 22:12:52,094 - DEBUG - Training subset scored 0.8124586956380595
Evaluating subsets:  84%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                     | 84/100 [03:12<00:49,  3.09s/it]2024-03-07 22:12:55,918 - DEBUG - Training subset scored 0.8766467131259746
2024-03-07 22:12:57,540 - DEBUG - Training subset scored 0.8774010931219514
2024-03-07 22:12:58,641 - DEBUG - Training subset scored 0.8525772019555748
2024-03-07 22:13:02,168 - DEBUG - Training subset scored 0.8612558525469657
Evaluating subsets:  88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                            | 88/100 [03:22<00:35,  2.92s/it]2024-03-07 22:13:04,027 - DEBUG - Training subset scored 0.8707492771531663
2024-03-07 22:13:04,276 - DEBUG - Training subset scored 0.8729583733114847
2024-03-07 22:13:07,822 - DEBUG - Training subset scored 0.8759446499047223
2024-03-07 22:13:09,914 - DEBUG - Training subset scored 0.8886317464660455
Evaluating subsets:  92%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                  | 92/100 [03:30<00:21,  2.63s/it]2024-03-07 22:13:10,132 - DEBUG - Training subset scored 0.8836322395163192
2024-03-07 22:13:13,730 - DEBUG - Training subset scored 0.8853636165589418
2024-03-07 22:13:15,371 - DEBUG - Training subset scored 0.8793252871530762
2024-03-07 22:13:16,329 - DEBUG - Training subset scored 0.8966828933724091
Evaluating subsets:  96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋         | 96/100 [03:36<00:09,  2.32s/it]2024-03-07 22:13:19,487 - DEBUG - Training subset scored 0.8761089015606255
2024-03-07 22:13:20,754 - DEBUG - Training subset scored 0.8697581984498367
2024-03-07 22:13:21,650 - DEBUG - Training subset scored 0.879255003658716
2024-03-07 22:13:25,137 - DEBUG - Training subset scored 0.8397256550886746
Evaluating subsets: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
2024-03-07 22:13:27,346 - DEBUG - Training subset scored 0.8541281186476043
2024-03-07 22:13:28,393 - DEBUG - Training subset scored 0.8298580040947826
2024-03-07 22:13:31,275 - DEBUG - Training subset scored 0.8708702459456279
2024-03-07 22:13:32,908 - DEBUG - Training subset scored 0.8796734928599856
2024-03-07 22:13:34,537 - DEBUG - Training subset scored 0.8723493016450448
2024-03-07 22:13:37,109 - DEBUG - Training subset scored 0.8610387542369908
2024-03-07 22:13:43,201 - DEBUG - Training subset scored 0.865791285427223
2024-03-07 22:13:43,208 - DEBUG - Best score: 0.8966828933724091 with subset: (({'title': 'Amazon.com: Highly Rated Miniature Toys Club – Amazon Subscribe & Discover, for Kids Ages 3 and up : Everything Else', 'h1': 'Highly Rated Miniature Toys Club – Amazon Subscribe & Discover, for Kids Ages 3 and up', 'product_copy': 'WHAT IS THIS CLUB? Miniature Toys Club is a monthly subscription that offers a new miniature toy, for kids ages 3 years and up, every month, right to your door. HIGHLY RATED PRODUCTS BY AMAZON CUSTOMERS: This club includes highly rated miniature toys that other Amazon customers have loved over the years. ALWAYS KNOW WHATS COMING NEXT: We will let you know what is coming next for every item, so you have the opportunity to skip a product that you do not want. HOW DOES THE SUBSCRIPTION WORK? You can subscribe with confidence as you can skip an item or cancel your subscription any time. To learn more, please check out the FAQs below. WHAT IS A SUBSCRIPTION CLUB AND HOW DOES IT DIFFER FROM A SUBSCRIPTION BOX? A subscription club sends you a single, new, and different item that matches your clubs theme at the frequency you select.'}, 'Highly-Rated-Miniature-Toys-Club'), ({'title': 'Amazon.com: Little Passports Science Jr. Premium - Subscription Box for Kids | Ages 5-8 : Everything Else', 'h1': 'Little Passports Science Jr. Premium - Subscription Box for Kids | Ages 5-8', 'product_copy': 'In the first month you’ll receive our weddell seals experiment kit, adventure-packed comic, Antarctica board game, and trading cards highlighting fun scientific facts. Your ongoing monthly packages contain everything you need to spark their natural curiosity, including hands-on science experiments, adventure comics, and trading cards highlighting fun scientific facts. Receive a curated book with each monthly package! Books are specially designed and hand-selected to match each month’s theme and nurture your child’s love of reading. Perfect for kids ages 5 - 8 Little Passports packages are designed in conjunction with professional educators, PhDs, and award-winning writers and designers.'}, 'Little-Passports-Science-Jr-Premium'))
Hit enter to evaluate the trained model...
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:04<00:00,  9.27it/s]
Before Training Accuracy: 0.8194524481637916
After Training Accuracy: 0.8744046199307406
(venv) aelaguiz@Amirs-MacBook-Pro langdspy % python examples/amazon/generate_slugs.py model.pkl
Hit enter to evaluate the trained model...
Predicting: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38/38 [00:03<00:00, 11.52it/s]
Before Training Accuracy: None
After Training Accuracy: 0.878812296826869
```