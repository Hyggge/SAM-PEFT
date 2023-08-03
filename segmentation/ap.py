def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    print(mrec)
    print(mpre)

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
        
    print(mpre)

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)
    print(i_list)

    ap = 0.0
    for i in i_list:
        print(mrec[i], mrec[i - 1], mpre[i])
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    print(ap)
    return ap, mrec, mpre


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    # iou_5 = result_stat[iou]

    tp = [1, 0, 0, 1, 0, 1, 1]
    fp = [0, 1, 1, 0, 1, 0, 0]
    assert len(fp) == len(tp)

    gt_total = 5

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    # return ap, mrec, mprec

# tp = [1, 0, 0, 1, 0, 1, 1]
# fp = [0, 1, 1, 0, 1, 0, 0]
# calculate_ap({0.5: {"fp": fp, "tp": tp}}, 0.5)

calculate_ap(None, None)
