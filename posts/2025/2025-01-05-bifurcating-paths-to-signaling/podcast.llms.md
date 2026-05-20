1.  Convert WAV to MP3

&nbsp;

    ffmpeg -i part1.wav -b:a 192k part1.mp3

2.  Update the HTML

&nbsp;

    <audio controls="1">
      <source src="part1.mp3" type="audio/mpeg">
    </audio>

    <link rel="alternate" type="application/rss+xml" title="Podcast RSS Feed" href="https://your-site.com/feed.xml">

    <?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
      <channel>
        <title>Your Podcast Title</title>
        <link>https://your-site.com</link>
        <description>A brief description of your podcast.</description>
        <language>en-us</language>
        <itunes:author>Your Name</itunes:author>
        <itunes:explicit>no</itunes:explicit>
        <itunes:image href="https://your-site.com/logo.jpg" />
        <item>
          <title>Episode 1: Title</title>
          <description>A brief description of the episode.</description>
          <link>https://your-site.com/episodes/episode1.mp3</link>
          <enclosure url="https://your-site.com/episodes/episode1.mp3" type="audio/mpeg" length="1234567" />
          <guid>https://your-site.com/episodes/episode1.mp3</guid>
          <pubDate>Thu, 28 Nov 2024 12:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman,
  author = {Bochman, Oren},
  url = {https://orenbochman.github.io/posts/2025/2025-01-05-bifurcating-paths-to-signaling/podcast.html},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. n.d. <https://orenbochman.github.io/posts/2025/2025-01-05-bifurcating-paths-to-signaling/podcast.html>.
