import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography'

const HomePage = () => {
    return (
        <Box sx={{ flexGrow: 1}}>
            <Grid container>
                <Grid item xs={6} sm={6} md={8} lg={6} xl={6}>
                    <Box sx={{
                        flexGrow: 1,
                        backgroundColor: 'primary.light',
                        padding: 10,
                    }}>
                        <Typography
                            variant='h2'
                            noWrap
                            component="div">
                            Project Title
                        </Typography>

                    </Box>
                </Grid>
                <Grid item xs={6} sm={6} md={8} lg={6} xl={6}>
                    <Box sx={{
                        flexGrow: 1,
                        backgroundColor: 'primary.light',
                        padding: 10
                    }}>
                        <Typography
                            variant='h2'
                            noWrap
                            component="div">
                            Image
                        </Typography>
                    </Box>
                </Grid>

                <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
                    <Box sx={{
                        flexGrow: 1,
                        backgroundColor: 'primary.light',
                        padding: 10
                    }}>
                        <Typography
                            variant='h4'
                            noWrap
                            component="div">
                            Project Description
                        </Typography>
                    </Box>
                </Grid>

                <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
                    <Box sx={{
                        flexGrow: 1,
                        backgroundColor: 'primary.light',
                        padding: 10
                    }}>
                        <Typography
                            variant='h4'
                            noWrap
                            component="div">
                            How to use the website
                        </Typography>
                    </Box>
                </Grid>
            </Grid>
        </Box>
    )
}

export default HomePage