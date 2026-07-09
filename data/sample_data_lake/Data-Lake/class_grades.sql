DROP TABLE IF EXISTS class_scores;

CREATE TABLE class_scores (
    student_id INTEGER PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL,
    class_name VARCHAR(20) NOT NULL,
    math_score DECIMAL(4,1) NOT NULL,
    literature_score DECIMAL(4,1) NOT NULL,
    english_score DECIMAL(4,1) NOT NULL
);

INSERT INTO class_scores
(student_id, student_name, class_name, math_score, literature_score, english_score)
VALUES
(1, 'Nguyen An', '10A1', 8.0, 7.5, 8.5),
(2, 'Tran Binh', '10A1', 6.5, 7.0, 7.5),
(3, 'Le Chi', '10A1', 9.0, 8.5, 9.0),
(4, 'Pham Dung', '10A1', 7.0, 6.5, 7.0),
(5, 'Hoang Giang', '10A1', 8.5, 8.0, 8.0),
(6, 'Do Ha', '10A1', 5.5, 6.0, 6.5),
(7, 'Bui Khanh', '10A1', 7.5, 7.5, 8.0),
(8, 'Vo Linh', '10A1', 9.5, 9.0, 8.5),
(9, 'Dang Minh', '10A1', 6.0, 6.5, 6.0),
(10, 'Mai Nhi', '10A1', 8.0, 8.5, 9.0);
