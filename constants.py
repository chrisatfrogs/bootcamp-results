COUNT_COLUMNS = ['action_coherence',
'citation_correct',
'grammar',
'factual_correctness',
'exhaustive_information',
'identical_information',
'lexical_correctness',
'syntactic_correctness']

LIKERT_COLUMNS = ['linguistic_difference', 'content_similarity', 'overall_quality']

COUNT_MAPPER = {k: f'{k}_percent' for k in COUNT_COLUMNS}
LIKERT_MAPPER = {k: f'{k}_percent' for k in LIKERT_COLUMNS}
INV_COUNT_MAPPER = {v: k for k, v in COUNT_MAPPER.items()}
INV_LIKERT_MAPPER = {v: k for k, v in LIKERT_MAPPER.items()}
ALL_PERCENT_COLUMNS = list(COUNT_MAPPER.values()) + list(LIKERT_MAPPER.values())

COMMENTS = {
    'anna.baumann': '''
    ### Comments
    There are **no significant differences** between the ratings of the two annotators.

    ```python
    s = "Python syntax highlighting"
    print(s)
    ```

    1. First ordered list item
    2. Another item
    ⋅⋅* Unordered sub-list. 
    1. Actual numbers don't matter, just that it's a number
    ⋅⋅1. Ordered sub-list
    4. And another item.

    ⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

    ⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
    ⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
    ⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

    * Unordered list can use asterisks
    - Or minuses
    + Or pluses
    
    ''',
    'anton.rothe': '''
    ''', 
    'emma.carballal-haire': '''
    ''',
    'erblina.morina': '''
    ''',  
    'larissa.rath': '''
    ''', 
    'theresa.herrmann': '''
    '''
}
