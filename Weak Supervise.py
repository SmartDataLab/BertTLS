def weak_supervision_selection(
    doc_sent_list, doc_date_list, doc_page_list, doc_taxoscore_list, abstract_size = 8, page_weight = 1, taxo_weight = 10, use_date = True, date_size = 2
):

    origin_tuple = tuple(zip(range(len(doc_page_list)), doc_page_list, doc_taxoscore_list, doc_date_list))
    
    #oracle选取初步想法：
    #1.先由page, taxo加权排序得到sorted_tuple(page越大越好，taxo越小越好), 
    #2.再取sorted_tuple中前n*abstract_size个(n待定)组成小集合selected_tuple
    #3.再对小集合里所有的timeline组合取'date方差'最小的一组得到result_tuple
    
    sorted_tuple = sorted(
        origin_tuple, key=lambda x: page_weight*x[1]+taxo_weight*(4-x[2]), reverse=True #w_page * page + w_taxo * (4-taxo)
    )
    
    if len(sorted_tuple) <= abstract_size: use_date=False #当总文章数小于或等于时间线size数时，无法使用date筛选
    
    if use_date == True:
        select_size = min(math.ceil(date_size*abstract_size), len(sorted_tuple))
        selected_tuple = sorted_tuple[0:select_size-1]
        min_Var_date = float("inf")
        for timeline in itertools.combinations(selected_tuple, abstract_size):
            dates = list(zip(*timeline))[3]
            datenum = [int(date[0:8]) for date in dates]
            Var_date = np.var(datenum)
            if Var_date < min_Var_date :
                min_Var_date = Var_date
                result_tuple = timeline
    
    else:
        result_tuple = sorted_tuple[0:min(abstract_size, len(sorted_tuple))]
        
        
    result_tuple = sorted(result_tuple,key = lambda x: x[3]) #result_tuple按时间顺序排序
    
    
    abstract = []
    abstract_date = []
    oracle_ids = []
    for i in range(len(result_tuple)):
        idx = result_tuple[i][0]
        oracle_ids.append(idx)
        abstract.append(doc_sent_list[idx])
        abstract_date.append(doc_date_list[idx])
    
    return abstract, abstract_date, oracle_ids