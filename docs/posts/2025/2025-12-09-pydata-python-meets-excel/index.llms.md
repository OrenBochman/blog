# An error occurred.

Unable to execute JavaScript.

> **TIP:**
>
> - Python drives modern data workflows, yet Excel remains the lingua franca of business. Many Python-based data teams struggle when the “last mile” of delivery still involves exporting results to Excel for business users. This talk explores practical ways for Python users to automate, scale, and enhance Excel-heavy processes using open-source libraries.
> - This talk will help you bridge the gap between code and the business-facing spreadsheet world.
> - We will discuss real-world use cases for report generation, batch processing, and dashboard templating, all from a Python-first perspective.

- This talk is designed for Python developers, analysts, and data scientists who routinely interact with Excel-based deliverables in their organization. It focuses on practical workflows that enhance productivity and reproducibility without requiring the audience to write or understand VBA or Excel formulas.
- The session begins by outlining common challenges Python users face when integrating with Excel, then introduces powerful Python tools that offer users seamless Excel file manipulation, specifically pandas, `xlsxwriter`, and `xlwings`.
- We will discuss some real-world use cases, such as generating reports, automating dashboards, creating custom functions in Excel and batch processing Excel files at scale.
- The talk concludes with a summary of tools, limitations, and best practices for integrating Python into Excel-centric workflows. This is a conceptual and strategic talk aimed at helping Python professionals work more effectively with Excel natives in the business ecosystem

[workshop repo](https://github.com/hugobowne/AI-for-SWEs)

> **TIP:**
>
> - Nisha Arora
>   - Dr. Nisha Arora is a data professional with experience across analytics, data science, reporting automation, storytelling, and applied statistical methods using Python, R, and Excel.
>   - With a background spanning technical writing, reviewing, and corporate trainings, she focuses on making advanced tools accessible to analysts and non-technical users.
>   - Her work bridges business-facing tools like Excel with scalable, reproducible workflows in Python. She creates accessible, practical learning content and actively contributes to the data community through her trainings, talks, and YouTube channel.
>   - She is currently working on a book project aimed at helping professionals modernize spreadsheet-based processes through Python.

## Outline

Hello everyone and welcome to PyData 2025. I’m excited to kick off the general track today with “Python Meets Excel: Smarter Workflows for Analysts and Data Teams” with Dr. Nisha Aurora. Please interact in the chat and drop questions in the Q&A.

[![Title](slide01.png)](slide01.png "Title")

Title

Welcome to the talk “Python Meets Excel: Smarter Workflows for Analysts and Data Teams.” I’m Dr. Nisha Aurora, a trainer and educator passionate about simplifying complex concepts for analysts and business users.

[![Bio](slide02.png)](slide02.png "Bio")

Bio

I have a PhD in Mathematics and have taught engineers, MBAs, and corporate teams. I love to speak at tech events and contribute to the community through blogs, forums, and YouTube.

[![Background](slide03.png)](slide03.png "Background")

Background

This talk is inspired by my upcoming book, “Python-Powered Excel,” which explores how Python and Excel can be integrated to create smarter workflows for analysts and data teams.

[![Upcoming Book](slide04.png)](slide04.png "Upcoming Book")

Upcoming Book

I love contributing to the community by writing blogs, answering questions on forums, and creating courses. My content has reached over 1.7 million users worldwide.

[![Community Contribution](slide05.png)](slide05.png "Community Contribution")

Community Contribution

Today’s agenda: - Why Python and Excel? - Tools for Python-Excel integration. - Case studies: Real-world examples. - Best practices and limitations.

[![Agenda](slide06.png)](slide06.png "Agenda")

Agenda

Why drag cells when Python can drive? Python is better for analytics, data science, and machine learning. But Excel is still the language of business and widely used by stakeholders.

[![Why drag cells when python can drive](slide07.png)](slide07.png "Why drag cells when python can drive")

Why drag cells when python can drive

Excel is everywhere. Business people understand and prefer Excel for its familiarity and flexibility. Deliverables are often expected in Excel format.

[![Excel isn’t going anywhere](slide08.png)](slide08.png "Excel isn’t going anywhere")

Excel isn’t going anywhere

Pandas is a powerful library for data analytics. It allows you to analyze data and export results to Excel, but formatting and customization require additional tools.

[![pandas](slide09.png)](slide09.png "pandas")

pandas

The ExcelWriter class in pandas enables customization. You can write multiple datasets to the same sheet and format headers, numbers, and dates.

[![ExcelWriter](slide10.png)](slide10.png "ExcelWriter")

ExcelWriter

Python meets Excel through various tools. Excel 365 introduced Python integration, but it has limitations like requiring an internet connection and limited library access.

[![Python meets Excel](slide11.png)](slide11.png "Python meets Excel")

Python meets Excel

Python in Excel is a good start for Excel users learning Python. However, it has limitations, such as restricted library access and reliance on Microsoft servers.

[![Python in Excel](slide12.png)](slide12.png "Python in Excel")

Python in Excel

Core Python libraries like pandas, NumPy, and Matplotlib are available in Excel 365. These libraries are essential for analytics and data visualization.

[![Core Python in Excel Libraries](slide13.png)](slide13.png "Core Python in Excel Libraries")

Core Python in Excel Libraries

Python tools for Excel include openpyxl, xlsxwriter, and xlwings. These libraries enable advanced Excel file manipulation and automation.

[![Python tools for Excel](slide14.png)](slide14.png "Python tools for Excel")

Python tools for Excel

Open-source Python libraries allow you to create charts, format data, and automate workflows in Excel, making it easier to deliver polished reports.

[![Open Source Python Libraries](slide15.png)](slide15.png "Open Source Python Libraries")

Open Source Python Libraries

With xlwings, you can use Excel as a user interface and Python as the backend engine, enabling seamless integration and automation.

[![What can you do with xlwings](slide16.png)](slide16.png "What can you do with xlwings")

What can you do with xlwings

Excel can serve as a user interface while Python acts as the engine. This approach combines the best of both tools for business and technical users.

[![Excel as UI & Python as Engine](slide17.png)](slide17.png "Excel as UI & Python as Engine")

Excel as UI & Python as Engine

Python can generate reports directly in Excel. This allows for automated, scalable, and reproducible workflows for analysts and data teams.

[![Report Generated by Python 1](slide18.png)](slide18.png "Report Generated by Python 1")

Report Generated by Python 1

Python-generated reports can include advanced formatting, charts, and dashboards, making them ready for business use.

[![Report Generated by Python 2](slide19.png)](slide19.png "Report Generated by Python 2")

Report Generated by Python 2

Let’s see Python and Excel integration in action. We’ll explore how to create automated workflows and dashboards.

[![Let’s see that in action](slide20.png)](slide20.png "Let’s see that in action")

Let’s see that in action

Python and Excel can connect seamlessly to create powerful, automated workflows for data analysis and reporting.

[![Connect](slide21.png)](slide21.png "Connect")

Connect

Thank you for attending “Python Meets Excel.” I hope this talk inspires you to explore Python-Excel integration for smarter workflows.

[![Thank You](slide22.png)](slide22.png "Thank You")

Thank You

Sneak peek: Python-powered Excel workflows can transform how analysts and data teams work, making processes more efficient and scalable.

[![Sneek Peek: Python–Powered Excel](slide23.png)](slide23.png "Sneek Peek: Python–Powered Excel")

Sneek Peek: Python–Powered Excel

Python-powered Excel workflows enable analysts to automate repetitive tasks, create dynamic dashboards, and scale their processes.

[![Sneek Peek: Python–Powered Excel](slide24.png)](slide24.png "Sneek Peek: Python–Powered Excel")

Sneek Peek: Python–Powered Excel

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Python {Meets} {Excel}},
  date = {2025-12-09},
  url = {https://orenbochman.github.io/posts/2025/2025-12-09-pydata-python-meets-excel/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Python Meets Excel.” December 9. <https://orenbochman.github.io/posts/2025/2025-12-09-pydata-python-meets-excel/>.
