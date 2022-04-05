import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography'
import Landing from "../assets/landing.jpg"

const HomePage = () => {
    return (
        <Box sx={{
            flexGrow: 1,
            backgroundColor: 'background.default'
        }}>
            <Grid container>
                <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
                    <div style={{
                        backgroundImage: `url(${Landing})`,
                        backgroundPosition: 'center',
                        backgroundRepeat: "no-repeat",
                        backgroundSize: 'cover',
                        minHeight: 700
                    }}>

                        <Typography
                            variant='h2'
                            component="div"
                            textAlign={'right'}
                            paddingTop={35}
                            paddingRight={5}>
                            Intent Identifcation
                        </Typography>
                        <Typography
                            variant='h5'
                            component="div"
                            textAlign={'right'}
                            paddingTop={2}
                            paddingRight={5}>
                            Recognize the intent behind spoken english sentences.
                        </Typography>
                    </div>
                </Grid>

            </Grid>
        </Box>
    )
}

export default HomePage