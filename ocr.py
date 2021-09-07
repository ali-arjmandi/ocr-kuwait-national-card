import cv2
import glob
import difflib
import numpy as np
import pandas as pd
import pytesseract
import sys


def out(dict):
    print(str(dict).replace('\'', '"'))


def eng_ocr(mask):
    try:
        eng_text = pytesseract.image_to_data(mask,
                                             config='--psm 6 -l eng -c textord_old_xheight=1 -c tessedit_char_blacklist=([!\\\"“.:])')
    except Exception as e:
        eng_text = None
        print(e)
    return eng_text


def ara_ocr(mask):
    try:
        ara_text = pytesseract.image_to_data(mask,
                                             config='--psm 6 -l ara -c textord_old_xheight=2 -c tessedit_char_blacklist=([!\\\"“.:])')
    except Exception as e:
        ara_text = None
        print(e)
    return ara_text


def gen_data_frame(text):
    row = text.split("\n")
    columns = row[0].split('\t')
    units = []
    for h in range(1, len(row) - 1):
        units.append(row[h].split('\t'))
    df = pd.DataFrame(units, columns=columns)
    df.iloc[:, :11] = df.iloc[:, :11] = df.iloc[:, :11].astype(int)
    return df


def calc_status(texts):
    #  0: moraje, 1: common, 2: passport
    status = 0
    for word in texts:
        seq = difflib.SequenceMatcher(None, word, "Passport")
        if seq.ratio() > 0.7:
            status = 2
    if not status:
        for word in texts:
            seq = difflib.SequenceMatcher(None, word, "Nationality")
            if seq.ratio() > 0.7:
                status = 1
    return status


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def show_img(*imgs):
    index = 1
    for img in imgs:
        cv2.imshow(str(index), img)
        index += 1
    cv2.waitKey(0)


if __name__ == "__main__":
    try:
        path = sys.argv
        if len(path) != 2:
            out({
                'status': '0',
                'error': 'args are not valid!'
            })
            exit()

        info_margin = 0.63
        left_margin = 0.245
        top_margin = 0.22
        addr = path[1]

        img = cv2.imread(addr)
        img = cv2.bilateralFilter(img, 35, 75, 75)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = apply_brightness_contrast(img, 10, 70)
        # img = cv2.GaussianBlur(img, (3, 3), 0)

        img_h, img_w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0.5)
        edge = cv2.Canny(blur, 0, 50, 3)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        # for con in contours:
        #     if cv2.contourArea(con) > 1000:
        #         print(cv2.contourArea(con))
        #     approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        approx = cv2.approxPolyDP(c, 0.009 * cv2.arcLength(c, True), True)
        approx = sorted(approx, key=lambda x: x[0][0])
        dots = []
        if len(approx) == 4:
            if approx[0][0][1] < approx[1][0][1]:
                dots.append(approx[0][0])
                dots.append(approx[1][0])
            else:
                dots.append(approx[1][0])
                dots.append(approx[0][0])
            if approx[2][0][1] < approx[3][0][1]:
                dots.append(approx[2][0])
                dots.append(approx[3][0])
            else:
                dots.append(approx[3][0])
                dots.append(approx[2][0])
            if dots[2][0]-dots[0][0] >= img_h * 0.6:
                input = np.float32([dots[0], dots[2], dots[3], dots[1]])
                output = np.float32(
                    [[0, 0], [img_w - 1, 0], [img_w - 1, img_h - 1], [0, img_h - 1]])
                matrix = cv2.getPerspectiveTransform(input, output)
                gray = cv2.warpPerspective(gray, matrix, (img_w, img_h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))
        gray = cv2.rectangle(
            gray, (0, 0), (int(img_w * left_margin), img_h), 255, -1)
        gray = cv2.rectangle(
            gray, (0, 0), (img_w, int(img_h * top_margin)), 255, -1)
        mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1,
                                sigmaY=1, borderType=cv2.BORDER_DEFAULT)
        mask = 255 - mask

        fx = 1
        fy = 1
        tmp_mask = cv2.resize(mask, None, fx=fx, fy=fy,
                              interpolation=cv2.INTER_CUBIC)
        eng_text = eng_ocr(tmp_mask)

        if eng_text:
            df = gen_data_frame(eng_text)
            df = df.loc[(df['conf'] != -1)]
            # 0: moraje, 1: common, 2: passport
            status = calc_status(df['text'])

            if status == 1:
                res = {
                    'type': 1,
                    'ar_name': None,
                    'en_name': None,
                    'ar_nationality': None,
                    'en_nationality': None,
                    'ar_sex': None,
                    'en_sex': None,
                    'birth': None,
                    'expiry': None,
                }
                # =========================================================================

                en_name_mask = tmp_mask[int(
                    img_h * 0.5 * fy):int(img_h * 0.71 * fy), int(img_w * 0.31 * fx): int(img_w * 0.9 * fx):]
                en_name_text = eng_ocr(en_name_mask)
                en_name = gen_data_frame(en_name_text)
                en_name = en_name.loc[(en_name['conf'] != -1)
                                      & (en_name['width'] > 70)]
                res['en_name'] = " ".join(en_name.text)
                en_sex_mask = tmp_mask[int(
                    img_h * 0.7 * fy):int(img_h * 0.85 * fy), int(img_w * 0.2 * fx):int(img_w * 0.6 * fx)]
                en_sex_mask = cv2.copyMakeBorder(
                    en_sex_mask, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                en_sex_text = pytesseract.image_to_data(en_sex_mask,
                                                        config='--psm 6 --oem 3 -l eng')
                en_sex_all = gen_data_frame(en_sex_text)
                en_sex_all = en_sex_all.loc[(
                    en_sex_all['conf'] != -1) & (en_sex_all['width'] > 10)]
                n_flag = 0
                s_flag = 0
                for i in en_sex_all.text:
                    if n_flag:
                        res['en_nationality'] = i
                        if i.lower() == "kwt":
                            res['ar_nationality'] = 'کویتی'
                        elif i.lower() == "sau":
                            res['ar_nationality'] = 'سعودی'

                        n_flag = 0
                    seq = difflib.SequenceMatcher(
                        None, "nationality", i.lower())
                    if seq.ratio() > 0.5 and not n_flag:
                        n_flag = 1
                    if s_flag:
                        if "f" in i.lower():
                            res['en_sex'] = "F"
                            res['ar_sex'] = "النثی"
                        if "m" in i.lower():
                            res['en_sex'] = "M"
                            res['ar_sex'] = "ذکر"
                        s_flag = 0
                    seq = difflib.SequenceMatcher(None, "sex", i.lower())
                    if seq.ratio() > 0.5 and not s_flag:
                        s_flag = 1

                # =========================================================================
                fx = 1.5
                fy = 1.11
                tmp_mask = cv2.resize(
                    mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                id_mask = tmp_mask[int(img_h * 0.2 * fy):int(img_h * 0.358 * fy),
                                   int(img_w * 0.414 * fx):int(img_w * 0.75 * fx)]
                date_mask = tmp_mask[int(
                    img_h * 0.80 * fy):int(img_h * 0.995 * fy), int(img_w * 0.46 * fx):int(img_w * 0.71 * fx)]
                id_text = eng_ocr(id_mask)
                res['expiry'] = date_df.iloc[1].text

                date_text = eng_ocr(date_mask)
                if id_text:
                    id_df = gen_data_frame(id_text)
                    id_df = id_df.loc[(id_df['conf'] != -1)
                                      & (id_df['width'] > 300)]
                    res['id'] = id_df.iloc[0].text

                if date_text:
                    date_df = gen_data_frame(date_text)
                    date_df = date_df.loc[(
                        date_df['conf'] != -1) & (date_df['width'] > 50)]
                    res['birth'] = date_df.iloc[0].text

                # =========================================================================
                fx = 1.17
                tmp_mask = cv2.resize(
                    mask, None, fx=fx, fy=1, interpolation=cv2.INTER_CUBIC)
                ar_name_mask = tmp_mask[int(
                    img_h * 0.3):int(img_h * 0.49), int(img_w * 0.25 * fx):int(img_w * 0.8 * fx)]
                ar_name_mask = cv2.copyMakeBorder(
                    ar_name_mask, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                ar_name_mask = 255 - ar_name_mask
                kernel = np.ones((4, 4), np.uint8)
                ar_name_mask = cv2.morphologyEx(
                    ar_name_mask, cv2.MORPH_OPEN, kernel)
                ar_name_mask = 255 - ar_name_mask
                ar_name_mask = cv2.GaussianBlur(
                    ar_name_mask, (0, 0), sigmaX=1.32, sigmaY=1.32, borderType=cv2.BORDER_DEFAULT)
                ar_name_text = ara_ocr(ar_name_mask)
                if ar_name_text:
                    ar_name_df = gen_data_frame(ar_name_text)
                    ar_name_df = ar_name_df.loc[(
                        ar_name_df['conf'] != -1) & (ar_name_df['width'] > 25)]
                    res['ar_name'] = ' '.join(ar_name_df.text)
                out(
                    {
                        'status': '1',
                        'data': res
                    }
                )

            elif status == 2:
                res = {
                    'type': 2,
                    'ar_name': None,
                    'en_name': None,
                    'passport': None,
                    'ar_nationality': None,
                    'en_nationality': None,
                    'ar_sex': None,
                    'en_sex': None,
                    'birth': None,
                    'expiry': None,
                }
                en_name_mask = tmp_mask[int(
                    img_h * 0.49):int(img_h * 0.645), int(img_w * 0.33):int(img_w * 0.9)]
                en_name_text = eng_ocr(en_name_mask)
                en_name = gen_data_frame(en_name_text)
                en_name = en_name.loc[(en_name['conf'] != -1)
                                      & (en_name['width'] > 50)]
                res['en_name'] = " ".join(en_name.text)
                en_sex_mask = tmp_mask[int(img_h * 0.675 * fy):int(img_h * 0.81 * fy),
                                       int(img_w * 0.2 * fx):int(img_w * 0.6 * fx)]
                en_sex_mask = cv2.copyMakeBorder(en_sex_mask, 100, 100, 100, 100, cv2.BORDER_CONSTANT,
                                                 value=[255, 255, 255])
                en_sex_text = pytesseract.image_to_data(en_sex_mask,
                                                        config='--psm 6 --oem 3 -l eng')
                en_sex_all = gen_data_frame(en_sex_text)
                en_sex_all = en_sex_all.loc[(en_sex_all['conf'] != -1)]
                n_flag = 0
                s_flag = 0
                for i in en_sex_all.text:
                    if n_flag:
                        res['en_nationality'] = i
                        n_flag = 0
                    seq = difflib.SequenceMatcher(
                        None, "nationality", i.lower())
                    if seq.ratio() > 0.7 and not n_flag:
                        n_flag = 1
                    if s_flag:
                        if "f" in i.lower():
                            res['en_sex'] = "F"
                            res['ar_sex'] = "النثی"
                        if "m" in i.lower():
                            res['en_sex'] = "M"
                            res['ar_sex'] = "ذکر"
                        s_flag = 0
                    if "sex" in i.lower() and not s_flag:
                        s_flag = 1
                ar_nationality_mask = tmp_mask[int(img_h * 0.675 * fy):int(img_h * 0.76 * fy),
                                               int(img_w * 0.61 * fx):int(img_w * 0.81 * fx)]
                ar_nationality_mask = cv2.copyMakeBorder(ar_nationality_mask, 100, 100, 100, 100, cv2.BORDER_CONSTANT,
                                                         value=[255, 255, 255])
                ar_nationality_mask = 255 - ar_nationality_mask
                kernel = np.ones((3, 3), np.uint8)
                ar_nationality_mask = cv2.morphologyEx(
                    ar_nationality_mask, cv2.MORPH_OPEN, kernel)
                ar_nationality_mask = 255 - ar_nationality_mask
                ar_nationality_mask = cv2.GaussianBlur(ar_nationality_mask, (0, 0), sigmaX=1, sigmaY=1,
                                                       borderType=cv2.BORDER_DEFAULT)
                ar_nationality_text = ara_ocr(ar_nationality_mask)
                if ar_nationality_text:
                    ar_nationality = gen_data_frame(ar_nationality_text)
                    ar_nationality = ar_nationality.loc[(
                        ar_nationality['conf'] != -1) & (ar_nationality['width'] > 80)]
                    ar_nationality = ' '.join(ar_nationality.text)
                    if ar_nationality == "مبصرى" or ar_nationality == "بصری":
                        ar_nationality = 'مصری'
                    res['ar_nationality'] = ar_nationality
                ar_name_mask = tmp_mask[int(
                    img_h * 0.28):int(img_h * 0.47), int(img_w * 0.264 * fx):int(img_w * 0.790 * fx)]
                ar_name_mask = cv2.copyMakeBorder(ar_name_mask, 100, 100, 100, 100,
                                                  cv2.BORDER_CONSTANT,
                                                  value=[255, 255, 255])
                ar_name_mask = 255 - ar_name_mask
                kernel = np.ones((3, 3), np.uint8)
                ar_name_mask = cv2.morphologyEx(
                    ar_name_mask, cv2.MORPH_OPEN, kernel)
                ar_name_mask = 255 - ar_name_mask
                ar_name_mask = cv2.GaussianBlur(ar_name_mask, (0, 0), sigmaX=2, sigmaY=2,
                                                borderType=cv2.BORDER_DEFAULT)
                ar_name_text = ara_ocr(ar_name_mask)
                if ar_name_text:
                    ar_name_df = gen_data_frame(ar_name_text)
                    ar_name_df = ar_name_df.loc[
                        (ar_name_df['conf'] != -1) & (ar_name_df['width'] > 80)]
                    ar_name = ' '.join(ar_name_df.text)
                    res['ar_name'] = ar_name
                # =========================================================================
                fx = 1.48
                fy = 1.11
                tmp_mask = cv2.resize(
                    mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                id_mask = tmp_mask[int(img_h * 0.2 * fy):int(img_h * 0.3 * fy),
                                   int(img_w * 0.414 * fx):int(img_w * 0.75 * fx)]
                date_mask = tmp_mask[int(img_h * 0.785 * fy):int(img_h * 0.955 * fy),
                                     int(img_w * 0.465 * fx):int(img_w * 0.7 * fx)]
                passport_mask = tmp_mask[int(img_h * 0.61 * fy):int(img_h * 0.71 * fy),
                                         int(img_w * 0.45 * fx):int(img_w * 0.72 * fx)]
                id_text = eng_ocr(id_mask)
                date_text = eng_ocr(date_mask)
                passport_text = eng_ocr(passport_mask)
                if id_text:
                    id_df = gen_data_frame(id_text)
                    id_df = id_df.loc[(id_df['conf'] != -1)
                                      & (id_df['width'] > 300)]
                    res['id'] = id_df.iloc[0].text
                if date_text:
                    date_df = gen_data_frame(date_text)
                    date_df = date_df.loc[(
                        date_df['conf'] != -1) & (date_df['width'] > 50)]
                    res['birth'] = date_df.iloc[0].text
                    res['expiry'] = date_df.iloc[1].text
                if passport_text:
                    passport_df = gen_data_frame(passport_text)
                    passport_df = passport_df.loc[(
                        passport_df['conf'] != -1) & (passport_df['width'] > 200)]
                    res['passport'] = passport_df.iloc[0].text
                out(
                    {
                        'status': '1',
                        'data': res
                    }
                )
            elif status == 0:
                res = {
                    'type': 0,
                    'id': None,
                    'ar_name': None,
                    'birth': None,
                    'Issue': None,
                    'expiry': None,
                }
                mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, 89, 20)
                mask = cv2.GaussianBlur(
                    mask, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
                tmp_mask = 255 - mask
                test_mask = tmp_mask[int(img_h * 0.3 * fy):int(img_h * 0.9 * fy),
                                     int(img_w * 0.3 * fx):int(img_w * 0.75 * fx)]
                test_text = eng_ocr(test_mask)
                if test_text:
                    test_df = gen_data_frame(test_text)
                    test_df = test_df.loc[(
                        test_df['conf'] != -1) & (test_df['width'] > 150)]
                    if len(list(test_df.iloc)) != 4:
                        out({
                            'status': '0',
                            'error': 'Cannot find type of card!'
                        })
                        exit()
                    res['id'] = test_df.iloc[0].text
                    res['birth'] = test_df.iloc[1].text
                    res['Issue'] = test_df.iloc[2].text
                    res['expiry'] = test_df.iloc[3].text
                ar_name_mask = tmp_mask[int(img_h * 0.42 * fy):int(img_h * 0.57 * fy),
                                        int(img_w * 0.3 * fx):int(img_w * 0.75 * fx)]
                ar_name_mask = cv2.copyMakeBorder(ar_name_mask, 100, 100, 100, 100,
                                                  cv2.BORDER_CONSTANT,
                                                  value=[255, 255, 255])
                ar_name_mask = 255 - ar_name_mask
                kernel = np.ones((3, 3), np.uint8)
                ar_name_mask = cv2.morphologyEx(
                    ar_name_mask, cv2.MORPH_OPEN, kernel)
                ar_name_mask = 255 - ar_name_mask
                ar_name_mask = cv2.GaussianBlur(ar_name_mask, (0, 0), sigmaX=1, sigmaY=1,
                                                borderType=cv2.BORDER_DEFAULT)
                ar_name_text = ara_ocr(ar_name_mask)
                if ar_name_text:
                    ar_name_df = gen_data_frame(ar_name_text)
                    ar_name_df = ar_name_df.loc[
                        (ar_name_df['conf'] != -1) & (ar_name_df['width'] > 20)]
                    ar_name = list(
                        filter(lambda x: x.isalpha(), ar_name_df.text))
                    ar_name = ' '.join(ar_name)
                    res['ar_name'] = ar_name
                out(
                    {
                        'status': '1',
                        'data': res
                    }
                )
    except IndexError as e:
        out({
            'status': '0',
            'error': 'cannot find some values inside this card!'
        })
        exit()
    except Exception as e:
        out({
            'status': '0',
            'error': str(e)
        })
        exit()
